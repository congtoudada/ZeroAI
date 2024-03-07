from zero.core.component.feature.launcher_comp import LauncherComponent

if __name__ == '__main__':
    launcher = LauncherComponent("conf/application-dev.yaml")
    launcher.start()
    launcher.update()

