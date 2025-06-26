import random
import time

from core.utils.colors import random_color, random_color_from_palette
from core.utils.time import delayed_execution
from extensions.control_gui.src.lib.gui import GUI, Category, Page
from extensions.control_gui.src.lib.widgets.buttons import Button, MultiStateButton
from extensions.control_gui.src.lib.widgets.number import DigitalNumberWidget
from extensions.control_gui.src.lib.widgets.sliders import SliderWidget, ClassicSliderWidget
from extensions.control_gui.src.lib.widgets.select import MultiSelectWidget
from extensions.control_gui.src.lib.widgets.dial import RotaryDialWidget
from extensions.control_gui.src.lib.widgets.text import TextWidget
from extensions.control_gui.src.lib.widgets.text_input import InputWidget
from core.utils.network.network import getHostIP


def main():
    host = getHostIP()
    app = GUI(id='gui', host=host, ws_port=8102, run_js=True)

    # First category
    category1 = Category(id='widgets',
                         name='Widgets',
                         icon='🤖',
                         )

    app.addCategory(category1)

    # Make the pages
    page_buttons = Page(id='buttons',
                        name='Buttons',
                        )

    page_inputs = Page(id='inputs',
                       name='Inputs', )

    page_data = Page(id='data',
                     name='Data', )

    page_iframe = Page(id='iframe',
                       name='IFrame', )

    page_groups = Page(id='groups',
                       name='Groups', )

    page_visualization = Page(id='visualization',
                              name='Visualization', )

    page_misc = Page(id='misc',
                     name='Misc', )

    category1.addPage(page_inputs, position=1)
    category1.addPage(page_buttons, position=2)
    category1.addPage(page_data)
    category1.addPage(page_iframe)
    category1.addPage(page_groups)
    category1.addPage(page_visualization)
    category1.addPage(page_misc)

    subcat1 = Category(id='subcat1',
                       name='Sub 1', )

    subcat1_page1 = Page(id='subcat1_page1',
                         name='Page 1', )
    subcat1.addPage(subcat1_page1)
    category1.addCategory(subcat1)

    subcat2 = Category(id='subcat2',
                       name='Sub 2', )
    subcat2_page1 = Page(id='subcat2_page1',
                         name='Page 2', )
    subcat2.addPage(subcat2_page1)

    button22 = Button(id='button22', text='Button 22', config={})
    subcat2_page1.addObject(button22, width=3, height=3)

    button22.callbacks.click.register(
        lambda *args, **kwargs: button22.update(color=[random.random(), random.random(), random.random(), 1], ))

    category1.addCategory(subcat2)

    subcat11 = Category(id='subcat11',
                        name='Sub 11', )
    subcat11_page1 = Page(id='subcat11_page1',
                          name='Page 1-1', )
    subcat11.addPage(subcat11_page1)
    subcat1.addCategory(subcat11)

    # ------------------------------------------------------------------------------------------------------------------
    # Buttons
    button_1 = Button(id='button1', text='Button 1', config={})
    page_buttons.addObject(button_1, width=2, height=2)

    button_2 = Button(id='button2', text='Color Change', config={'color': [1, 0, 0, 0.2], 'fontSize': 16})
    page_buttons.addObject(button_2, column=3, width=4, height=4)

    button_2.callbacks.click.register(
        lambda *args, **kwargs: button_2.update(color=[random.random(), random.random(), random.random(), 1], ))

    button3 = Button(id='button3', text='Small Text', config={'color': "#274D27", 'fontSize': 10})
    page_buttons.addObject(button3, row=1, column=7, width=4, height=1)

    # Multi-State Button

    def msb_callback(button: MultiStateButton, *args, **kwargs):
        new_colors = []
        for color in button.config['color']:
            new_colors.append([random.random(), random.random(), random.random(), 1])
        button.update(color=new_colors)

    msb1 = MultiStateButton(id='msb1', states=['A', 'B', 'C'], color=['#4D0E11', '#0E4D11', '#110E4D'],
                            config={'fontSize': 16})
    # msb1.callbacks.state.register(msb_callback)
    page_buttons.addObject(msb1, row=2, column=12, width=2, height=2)

    msb2 = MultiStateButton(id='msb2', states=['State 1', 'State 2', 'State 3', 'State 4', 'State 5'],
                            color=[random_color() for _ in range(5)], title='Multi-State Button')

    page_buttons.addObject(msb2, row=6, column=12, width=4, height=2)

    def reset_button(button, *args, **kwargs):
        if button.state == 'ON':
            delayed_execution(lambda: button.update(state='OFF'), delay=5)

    msb3 = MultiStateButton(id='msb3', states=['OFF', 'ON'],
                            color=[[0.4, 0, 0], [0, 0.4, 0]], title='Reset')
    msb3.callbacks.state.register(reset_button)
    page_buttons.addObject(msb3, row=2, column=15, width=2, height=2)

    # ------------------------------------------------------------------------------------------------------------------
    # Sliders
    slider1 = SliderWidget(widget_id='slider1', min_value=0,
                           max_value=1,
                           increment=0.1,
                           value=0.5,
                           color=random_color_from_palette('dark'),
                           continuousUpdates=True,
                           automaticReset=0.5)
    page_inputs.addObject(slider1, height=2, width=5)

    slider2 = SliderWidget(widget_id='slider2',
                           min_value=0,
                           max_value=100,
                           increment=0.1,
                           value=20,
                           color=random_color_from_palette('dark'),
                           direction='vertical',
                           ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], )

    page_inputs.addObject(slider2, height=6, width=2, column=7)

    classic_slider_1 = ClassicSliderWidget(widget_id='cslider1',
                                           value=50,
                                           increment=25,
                                           backgroundColor=random_color_from_palette('dark'),
                                           title_position='left',
                                           valuePosition='right')
    page_inputs.addObject(classic_slider_1, width=8, height=1, row=8)

    msw1 = MultiSelectWidget(widget_id='msw1',
                             options={
                                 'optionA': {
                                     'label': 'A',
                                     'color': random_color_from_palette('dark'),
                                 },
                                 'optionB': {
                                     'label': 'B',
                                 },
                                 'optionC': {
                                     'label': 'C',
                                     'color': random_color_from_palette('dark'),
                                 }
                             },
                             title='Multi-Select',
                             value='optionA')

    page_inputs.addObject(msw1, column=11, width=6, height=3)

    msw2 = MultiSelectWidget(widget_id='msw2',
                             options={
                                 'optionA': {
                                     'label': 'Option A',
                                     'color': random_color_from_palette('dark'),
                                 },
                                 'optionB': {
                                     'label': 'Option B',
                                 },
                                 'optionC': {
                                     'label': 'Option C',
                                     'color': random_color_from_palette('dark'),
                                 }
                             },
                             title='Multi-Select',
                             title_position='left',
                             value='optionA')

    page_inputs.addObject(msw2, column=11, row=5, width=7, height=1)

    dial1 = RotaryDialWidget(widget_id='dial1', value=25, ticks=[0, 25, 50, 75, 100], limitToTicks=True,
                             )

    page_inputs.addObject(dial1,
                          column=20,
                          width=2,
                          height=3,
                          )

    dial2 = RotaryDialWidget(widget_id='dial2',
                             min_value=0,
                             max_value=1,
                             increment=0.05,
                             title_position='left',
                             value=0.5,
                             continuousUpdates=True,
                             dialColor=random_color_from_palette('pastel'),
                             dialWidth=8,
                             )

    page_inputs.addObject(dial2, column=20, row=5, width=4, height=3, )

    text_input_1 = InputWidget(widget_id='text_input_1')
    page_inputs.addObject(text_input_1, row=2, column=27, width=10, height=4)

    text_input_2 = InputWidget(widget_id='text_input_2',
                               title='Test:',
                               title_position='left',
                               color=random_color_from_palette('dark'),
                               datatype='int',
                               value=13,
                               tooltip="Integer",
                               validator=lambda x: x < 20)

    page_inputs.addObject(text_input_2, row=7, column=27, width=10, height=2)

    text_input_3 = InputWidget(widget_id='text_input_3',
                               title='Input 1:',
                               title_position='left',
                               inputFieldWidth="100px",
                               inputFieldPosition="right", )

    page_inputs.addObject(text_input_3, row=10, column=27, width=8, height=1)

    text_input_4 = InputWidget(widget_id='text_input_4',
                               title='Input 2:',
                               title_position='left',
                               inputFieldWidth="100px",
                               inputFieldPosition="right",
                               color=random_color_from_palette('dark'), )

    page_inputs.addObject(text_input_4, row=11, column=27, width=8, height=1)
    text_input_4.setValue("HALLO")

    # ==================================================================================================================
    # Data Page
    dnw1 = DigitalNumberWidget(widget_id='dnw1',
                               title='Theta',
                               value=10,
                               min_value=-1000,
                               max_value=1000,
                               increment=0.01,
                               color='transparent',
                               text_color=random_color_from_palette('pastel'),
                               value_color=[1, 1, 1]
                               )

    page_data.addObject(dnw1, width=5, height=1)

    text_widget_1 = TextWidget(widget_id='text_widget_1',
                               title='Text Widget',
                               text="Hallo 1 \n13\nThis is a third line",
                               horizontal_alignment='left',
                               vertical_alignment='top',
                               text_color=random_color_from_palette('pastel'),
                               font_weight='bold',
                               font_style='italic', )
    page_data.addObject(text_widget_1, width=5, height=5)
    # ==================================================================================================================
    category2 = Category(id='cat2',
                         name='Category 2', )

    app.addCategory(category2)

    # ==================================================================================================================
    i = 0
    while True:
        # new_button = Button(id=f'nbutton{i}', text=f'B {i}',
        #                     config={'color': random_color_from_palette('pastel'), 'text_color': [0, 0, 0]})
        # page_buttons.addObject(new_button, width=random.randint(1, 4), height=random.randint(1, 4))
        dnw1.value = random.randint(-10000, 10000) / 100
        i += 1
        time.sleep(0.1)


if __name__ == '__main__':
    main()
