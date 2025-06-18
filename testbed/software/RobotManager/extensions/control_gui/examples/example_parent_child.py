import time

from core.utils.network.network import getHostIP
from extensions.control_gui.src.lib.gui import GUI, ControlGUI_Category, ControlGUI_Page
from extensions.control_gui.src.lib.widgets.buttons import Button


def create_parent_gui(host, port):
    parent_app = GUI(id='parent_gui', host=host, ws_port=port)

    cat1 = parent_app.addCategory(ControlGUI_Category(id='category1', name='Category 1'))
    cat2 = parent_app.addCategory(ControlGUI_Category(id='category2', name='Category 2'))

    print(cat1.uid)
    page1 = cat1.addPage(ControlGUI_Page(id='page1', name='Page 1'))
    button1 = page1.addObject(Button(id='button1', text='Button 1', color=[0.3, 0, 0]))

    return parent_app


def create_child_gui(host, port, parent_port):
    child_app = GUI(id='child_gui', host=host, ws_port=port)

    cat1 = child_app.addCategory(ControlGUI_Category(id='category1', name='Child Cat 1'))
    cat2 = child_app.addCategory(ControlGUI_Category(id='category2', name='Child Cat 2'))
    page1 = cat1.addPage(ControlGUI_Page(id='page1', name='Child Page 1'))
    button1 = page1.addObject(Button(id='button1', text='CB1', color=[0, 0.6, 0]))

    # child_app.connectToParent(host, parent_port)

    return child_app


def main():
    host = getHostIP()
    parent_port = 8100
    child_port = 8101

    parent_app = create_parent_gui(host, parent_port)
    child_app = create_child_gui(host, child_port, parent_port)

    parent_app.connectToChild(id='child1', child_address=host, child_port=child_port)
    #
    # obj = parent_app.getElementByUID('/parent_gui::category1')
    # print(obj)


    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
