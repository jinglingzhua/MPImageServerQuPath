#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
独立运行的 WebP/AVIF OME-TIFF tile 服务 (不依赖 ModelZoo / PyOpenSlide).

把 convert2ometiff.py 产出的 webp / avif 金字塔 tif 按 tile 实时转码为
JPEG, 配合 QuPath 的 mpwebp 扩展实现 File -> Open URI 懒加载打开
(Choose file 选择本地文件, 或直接粘贴本机绝对路径).

依赖: python 标准库 + tifffile + imagecodecs + numpy

用法:
    直接运行(或双击打包后的 mpimg_tile_server) = 托盘模式:
        系统托盘显示图标, 右键菜单「退出」; 服务默认 127.0.0.1:8807
    /home/zmf/local/miniconda3/envs/mm/bin/python mpimg_tile_server.py \
        --console [--host 0.0.0.0] [--port 8807] [--quality 85]
        命令行模式(前台运行, 调试用)

路由 (全部 GET):
    /mpwebp/meta?path=<url编码的本机绝对路径>   -> json 元数据 (含 id)
    /mpwebp/tile/<id>/<level>/<y>/<x>.jpg      -> 该级该位置 tile 的 jpeg
    /mpwebp/label/<id>.jpg                     -> label 图
    /mpwebp/macro/<id>.jpg                     -> macro 图

meta 会校验首页压缩标签为 WEBP(50001) 或 AVIF(48), 其余 tif 返回 404,
QuPath 端据此放行给 Bio-Formats.
'''
import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse, parse_qs

import imagecodecs
import numpy as np
import tifffile

COMPRESSION_WEBP = 50001
COMPRESSION_AVIF = 48     # AVIF-in-TIFF 使用的压缩码(实测该写入方)

SUPPORTED_COMPTYPES = {COMPRESSION_WEBP: 'webp', COMPRESSION_AVIF: 'avif'}


class NotWebpError(RuntimeError):
    pass


class LRU:
    def __init__(self, cap=4096):
        self.cap = cap
        self.d = OrderedDict()
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            v = self.d.pop(k, None)
            if v is not None:
                self.d[k] = v
            return v

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            while len(self.d) > self.cap:
                self.d.popitem(last=False)


class WebpTifIndex:
    '''一个 webp ome-tif 的 tile 偏移索引 (TiffFile 保持打开)'''

    def __init__(self, path, jpeg_quality=85):
        self.path = os.path.abspath(path)
        self.id = hashlib.md5(self.path.encode()).hexdigest()[:12]
        st = os.stat(self.path)
        self.fstat = (st.st_mtime_ns, st.st_size)   # 用于索引失效判断
        self.jpeg_quality = jpeg_quality
        self.lock = threading.Lock()          # TiffPage/filehandle 均非线程安全
        self.cache = LRU(4096)
        self._open()

    def _open(self):
        self.tif = tifffile.TiffFile(self.path)
        page0 = self.tif.pages[0]
        comp = getattr(page0, 'compression', 0)
        comp_type = SUPPORTED_COMPTYPES.get(int(comp))
        if comp_type is None:
            self.tif.close()
            raise NotWebpError(
                f'{self.path}: 首页压缩为 {comp}, 不是 webp/avif tif')
        self.codec = comp_type
        if page0.tilewidth is None:
            raise NotWebpError(f'{self.path}: 不是 tiled tif')
        self.tile_w = page0.tilewidth
        self.tile_h = page0.tilelength
        self.width = int(page0.shape[1])
        self.height = int(page0.shape[0])

        # 各级分辨率: level0 + subifds(文件偏移, 须先 seek 再构造 TiffPage)
        self.levels = [page0]
        with self.lock:
            for lv, off in enumerate(page0.subifds or ()):
                self.tif.filehandle.seek(off)
                self.levels.append(tifffile.TiffPage(self.tif, (0, lv + 1)))

        # label / macro: 顶层小页 (subfiletype 1=label, 9=macro)
        self.label_jpeg = None
        self.macro_jpeg = None
        for p in self.tif.pages[1:]:
            sft = getattr(p, 'subfiletype', 0) or 0
            desc = p.description or ''
            try:
                if sft == 9 or 'macro' in desc:
                    self.macro_jpeg = imagecodecs.jpeg_encode(
                        p.asarray(), level=self.jpeg_quality)
                elif sft == 1 or 'label' in desc:
                    self.label_jpeg = imagecodecs.jpeg_encode(
                        p.asarray(), level=self.jpeg_quality)
            except Exception as e:
                print(f'warn: read associated image failed: {e}')

        xml = page0.description or ''
        self.mpp_x = self._parse_mpp(xml, 'X')
        self.mpp_y = self._parse_mpp(xml, 'Y')
        self.mag = None
        m = re.search(r'Magnification="([\d.]+)"', xml)
        if m:
            self.mag = float(m.group(1))

    @staticmethod
    def _parse_mpp(xml, axis):
        m = re.search(rf'PhysicalSize{axis}="([\d.eE+-]+)"', xml)
        return float(m.group(1)) if m else None

    def meta_json(self):
        return {
            'id': self.id,
            'codec': self.codec,
            'name': os.path.basename(self.path),
            'path': self.path,
            'width': self.width, 'height': self.height,
            'tileWidth': self.tile_w, 'tileHeight': self.tile_h,
            'levels': [{'w': int(p.shape[1]), 'h': int(p.shape[0])}
                       for p in self.levels],
            'mppX': self.mpp_x, 'mppY': self.mpp_y,
            'magnification': self.mag,
            'hasLabel': self.label_jpeg is not None,
            'hasMacro': self.macro_jpeg is not None,
        }

    def read_tile_jpeg(self, level, ty, tx):
        if not 0 <= level < len(self.levels):
            raise IndexError(f'bad level {level}')
        page = self.levels[level]
        rows = int(np.ceil(page.shape[0] / self.tile_h))
        cols = int(np.ceil(page.shape[1] / self.tile_w))
        if not (0 <= ty < rows and 0 <= tx < cols):
            raise IndexError(
                f'tile ({ty},{tx}) 超出网格 ({rows}x{cols}, level {level})')
        key = (level, ty, tx)
        data = self.cache.get(key)
        if data is not None:
            return data
        idx = ty * cols + tx
        with self.lock:
            fh = self.tif.filehandle
            fh.seek(page.dataoffsets[idx])
            raw = fh.read(page.databytecounts[idx])
        if self.codec == 'avif':
            rgb = imagecodecs.avif_decode(raw)
        else:
            rgb = imagecodecs.webp_decode(raw)
        data = imagecodecs.jpeg_encode(np.ascontiguousarray(rgb),
                                       level=self.jpeg_quality)
        self.cache.put(key, data)
        return data

    def close(self):
        try:
            self.tif.close()
        except Exception:
            pass


def _index_current(idx) -> bool:
    '''文件未被重新生成(mtime/size 未变)才认为索引有效'''
    try:
        st = os.stat(idx.path)
        return (st.st_mtime_ns, st.st_size) == idx.fstat
    except OSError:
        return False


class Registry:
    '''按本机绝对路径的索引表 (id = 路径 md5 前 12 位)'''

    def __init__(self, quality):
        self.quality = quality
        self.lock = threading.Lock()
        self.by_id = {}

    def index_path(self, path) -> WebpTifIndex:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'文件不存在: {path}')
        key = hashlib.md5(path.encode()).hexdigest()[:12]
        with self.lock:
            idx = self.by_id.get(key)
        if idx is not None and _index_current(idx):
            return idx
        # 文件不存在或已被重新生成(mtime/size 变化) -> 重建索引
        idx = WebpTifIndex(path, self.quality)   # idx.id == key
        with self.lock:
            self.by_id[idx.id] = idx
        return idx

    def get(self, key):
        with self.lock:
            idx = self.by_id.get(key)
        if idx is None:
            return None
        if not _index_current(idx):
            return self.index_path(idx.path)     # 文件已重新生成, 重建索引
        return idx


class MpWebpHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'MpWebpTileServer/1.2'

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, msg):
        # 状态行只支持 latin-1, 中文等错误详情放 UTF-8 JSON body
        self._send(code, json.dumps({'error': str(msg)}, ensure_ascii=False)
                   .encode('utf-8'), 'application/json')

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)
        parts = [p for p in path.split('/') if p]
        try:
            if not parts or parts[0] != 'mpwebp':
                self._send(200, __doc__.encode(), 'text/plain')
                return
            kind = parts[1]
            if kind == 'meta':
                if 'path' not in qs:
                    self._send_error(404, 'missing ?path=<本机绝对路径>')
                    return
                idx = self.server.registry.index_path(qs['path'][0])
                self._send(200, json.dumps(idx.meta_json()).encode(),
                           'application/json')
            elif kind == 'tile' and len(parts) == 6:
                idx = self.server.registry.get(parts[2])
                if idx is None:
                    self._send_error(404, f'unknown id: {parts[2]}')
                    return
                level, ty = int(parts[3]), int(parts[4])
                tx = int(parts[5].rsplit('.', 1)[0])
                self._send(200, idx.read_tile_jpeg(level, ty, tx),
                           'image/jpeg')
            elif kind in ('label', 'macro') and len(parts) == 3:
                key = parts[2][:-4] if parts[2].endswith('.jpg') else parts[2]
                idx = self.server.registry.get(key)
                if idx is None:
                    self._send_error(404, f'unknown id: {key}')
                    return
                data = getattr(idx, f'{kind}_jpeg')
                if data is None:
                    self._send_error(404, f'no {kind} image')
                    return
                self._send(200, data, 'image/jpeg')
            else:
                self._send_error(404, 'unknown route')
        except NotWebpError as e:
            self._send_error(404, str(e))
        except FileNotFoundError as e:
            self._send_error(404, str(e))
        except (IndexError, ValueError) as e:
            self._send_error(400, str(e))
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send_error(500, repr(e))
            except Exception:
                pass

    def log_message(self, fmt, *args):
        print(f'[{self.log_date_time_string()}] {self.address_string()} '
              f'{fmt % args}')


LOG_PATH = os.path.join(os.path.expanduser('~/.cache'),
                        'mpimg_tile_server.log')


def setup_logging():
    '''--windowed 打包下 stdout/stderr 为 None, 统一重定向到日志文件'''
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if sys.stdout is None or sys.stderr is None:
        sys.stdout = open(LOG_PATH, 'a', buffering=1)
        sys.stderr = sys.stdout


def already_running(port, host='127.0.0.1'):
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


class ServerThread(threading.Thread):
    def __init__(self, args):
        super().__init__(daemon=True)
        self.httpd = ThreadingHTTPServer((args.host, args.port), MpWebpHandler)
        self.httpd.daemon_threads = True
        self.httpd.registry = Registry(args.quality)

    def run(self):
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def serve_console(args):
    setup_logging()
    httpd = ThreadingHTTPServer((args.host, args.port), MpWebpHandler)
    httpd.daemon_threads = True
    httpd.registry = Registry(args.quality)
    print(f'MpWebp tile server: http://{args.host}:{args.port}/mpwebp/')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def serve_tray(args):
    setup_logging()
    if already_running(args.port):
        print(f'server already running on :{args.port}')
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication([])
            QMessageBox.information(None, 'MPImg tile server',
                                    f'服务已在运行 (端口 {args.port})')
        except Exception:
            pass
        return

    server = ServerThread(args)
    server.start()
    print(f'MpWebp tile server: http://{args.host}:{args.port}/mpwebp/')

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPixmap, QPainter
    from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor('#2b6cb0'))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor('white'))
    f = QFont()
    f.setBold(True)
    f.setPixelSize(26)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignCenter, 'MP')
    p.end()

    tray = QSystemTrayIcon(QIcon(pm))
    tray.setToolTip(f'MPImg tile server :{args.port}')

    menu = QMenu()
    info = menu.addAction(f'服务运行中 :{args.port}')
    info.setEnabled(False)
    menu.addSeparator()
    act_quit = QAction('退出', menu)
    act_quit.triggered.connect(app.quit)
    menu.addAction(act_quit)
    tray.setContextMenu(menu)
    tray.show()

    app.aboutToQuit.connect(server.stop)
    app.exec()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1',
                    help='托盘模式默认仅本机, 需局域网访问用 0.0.0.0')
    ap.add_argument('--port', type=int, default=8807)
    ap.add_argument('--quality', type=int, default=85,
                    help='tile 转码 jpeg 质量')
    ap.add_argument('--console', action='store_true',
                    help='命令行模式(前台打印日志); 默认托盘模式')
    args = ap.parse_args()

    if args.console:
        serve_console(args)
    else:
        serve_tray(args)


if __name__ == '__main__':
    main()
