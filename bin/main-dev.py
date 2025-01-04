import argparse
import traceback

from zero.core.launch_comp import LaunchComponent


def make_parser():
    parser = argparse.ArgumentParser("ZeroAI Demo!")
    parser.add_argument("-app", "--application", type=str, default="conf/application-dev.yaml")
    return parser


if __name__ == '__main__':
    # 解析args
    args = make_parser().parse_args()
    comp = LaunchComponent(args.application)
    try:
        comp.start()
        comp.update()
    except KeyboardInterrupt:
        comp.esc_event.set()
        comp.on_destroy()
    except Exception as e:
        print(f"LaunchComponent: {e}")
        print(traceback.format_exc())  # 打印完整的堆栈信息
        exit(1)
    exit(0)

