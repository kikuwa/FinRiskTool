import multiprocessing
import os

from app import create_app

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Windows spawn 子进程会 re-import 本模块；仅真正的服务主进程启动 Flask
    if multiprocessing.parent_process() is None:
        app = create_app()
        port = int(os.environ.get('PORT', 5005))
        # Windows + multiprocessing：热重载会导致子进程 orphaned，停止/taskkill 可能失效
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
