# -*- coding: utf-8 -*-
"""文本输入客户端（调试用）：连 voice_dialogue.py 的 --text-input-port，while input() 逐行发送。

用法：
    终端 1（主程序，文本输入端口默认关——起了端口本脚本才能连）：
        python examples/voice_dialogue.py --asr-device cuda --text-input-port 9123 ...
    终端 2（本脚本，默认端口 9123）：
        python examples/text_input.py [端口]

每输入一行 → 即时发给主程序（非阻塞：发完可继续输入下一行，不等回答）。
语义（文本输入模式）：
- 唤醒词/退出词**无效**——当普通句子送 LLM/agent，不触发唤醒/退出状态机；
- 打断词（默认"停下"）**整行完全等于** → 立即停当前输出，该行不进对话；无输出在途=无动作；
- 休眠态直接输入问题 → 自动唤醒直接对话（不播就绪语）；
- 与麦克风语音**并存**，两边输入都有效。
空行忽略。Ctrl+D / Ctrl+C 退出。主程序没起/断开 → 提示并重连。
"""

import socket
import sys
import time


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9123
    print("文本输入客户端：连 127.0.0.1:%d（主程序需 --text-input-port %d 才接受连接）"
          % (port, port), flush=True)
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError as e:
            print("连不上主程序（%s）——请先起 voice_dialogue.py --text-input-port %d；3 秒后重试…"
                  % (e, port), flush=True)
            time.sleep(3)
            continue
        print("已连接，逐行输入（Ctrl+D/Ctrl+C 退出；空行跳过；打断词='停下' 整行即停当前输出）",
              flush=True)
        try:
            while True:
                line = input("你> ")
                if not line.strip():
                    continue
                s.sendall((line + "\n").encode("utf-8"))
        except EOFError:
            print("\n退出。", flush=True)
            break
        except KeyboardInterrupt:
            print("\n退出。", flush=True)
            break
        finally:
            try:
                s.close()
            except Exception:
                pass
        # 主程序断开（如重启）→ 回到重连循环
        print("连接已断开，尝试重连…", flush=True)


if __name__ == "__main__":
    main()
