from zero.core.component.base.component import Component


class BaseHelperComponent(Component):
    def __init__(self, shared_data):
        super().__init__(shared_data)

    def update(self):
        if self.enable:
            self.on_update()
        # if self.esc_event.is_set():  # Helper生死交给外部
        #     self.destroy()
