import argparse

from zero.core.component.feature.launcher_comp import LauncherComponent


def make_parser():
    parser = argparse.ArgumentParser("ZeroAI Demo!")
    parser.add_argument("-app", "--application", type=str, default="conf/application-dev.yaml")
    return parser


if __name__ == '__main__':
    # 解析args
    args = make_parser().parse_args()
    launcher = LauncherComponent(args.application)
    launcher.start()
    launcher.update()

