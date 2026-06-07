import numpy as np
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.lang import Builder
import webbrowser

# ==========================
# MODEL SECTION
# ==========================

MEAN = np.array([
    2013.67083333,
    7.51201250,
    37508.55833333,
    1.17916667,
    0.64166667,
    0.12500000,
    0.05000000
], dtype=np.float32)

SCALE = np.array([
    2.87879882,
    8.97299670,
    41765.0649,
    0.404638899,
    0.479510746,
    0.330718914,
    0.269258240
], dtype=np.float32)


MODEL = np.load("car_price_numpy_weights.npz")

def relu(x):
    return np.maximum(x, 0)

def load_model():
    pass


def get_car_price_prediction(
    year,
    present_price,
    kms_driven,
    fuel_petrol,
    seller_individual,
    transmission_manual
):
    load_model()

    owner = 0

    x = np.array([
        year,
        present_price,
        kms_driven,
        owner,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ], dtype=np.float32)

    x = (x - MEAN) / SCALE
    x = x.reshape(1, 7).astype(np.float32)

    x = relu(x @ MODEL["dense_24_kernel"] + MODEL["dense_24_bias"])
    x = relu(x @ MODEL["dense_25_kernel"] + MODEL["dense_25_bias"])
    x = relu(x @ MODEL["dense_26_kernel"] + MODEL["dense_26_bias"])
    x = relu(x @ MODEL["dense_27_kernel"] + MODEL["dense_27_bias"])
    x = relu(x @ MODEL["dense_28_kernel"] + MODEL["dense_28_bias"])
    x = x @ MODEL["dense_29_kernel"] + MODEL["dense_29_bias"]

    base_price = float(x[0][0])

    final_price_toman = base_price * 100

    if 2018 <= year <= 2020:
        final_price_toman -= 20
    elif 2016 <= year < 2018:
        final_price_toman -= 25
    elif 2014 <= year < 2016:
        final_price_toman -= 35
    elif 2012 <= year < 2014:
        final_price_toman -= 45
    elif 2008 <= year < 2012:
        final_price_toman -= 60
    elif 2004 <= year < 2008:
        final_price_toman -= 80
    elif 2000 <= year < 2004:
        final_price_toman -= 100

    max_allowed = (present_price * 100) * 0.90

    if final_price_toman > max_allowed:
        final_price_toman = max_allowed

    return final_price_toman


KV = '''
MDScreen:
    md_bg_color: 0.95, 0.98, 1, 1

    MDBoxLayout:
        orientation: 'vertical'
        padding: "20dp"
        spacing: "10dp"

        MDLabel:
            text: "CAR PRICE EXPERT"
            halign: "center"
            font_style: "H4"
            bold: True
            theme_text_color: "Custom"
            text_color: 0.1, 0.4, 0.7, 1
            size_hint_y: None
            height: "50dp"

        MDCard:
            orientation: "vertical"
            padding: "15dp"
            radius: [25,]
            md_bg_color: 1, 1, 1, 1
            elevation: 1

            ScrollView:
                MDBoxLayout:
                    orientation: 'vertical'
                    adaptive_height: True
                    spacing: "12dp"
                    padding: [10, 15]

                    MDTextField:
                        id: year
                        hint_text: "Manufacturing Year"
                        mode: "rectangle"

                    MDTextField:
                        id: price
                        hint_text: "Present Price"
                        helper_text: "Enter price in units (e.g., 6.5 for 650M)"
                        helper_text_mode: "on_focus"
                        mode: "rectangle"

                    MDTextField:
                        id: kms
                        hint_text: "Total Kilometers"
                        mode: "rectangle"

                    MDTextField:
                        id: fuel
                        hint_text: "Fuel (0:Gas, 1:Pet, 2:Dsl)"
                        mode: "rectangle"

                    MDTextField:
                        id: seller
                        hint_text: "Seller (0:Dlr, 1:Ind)"
                        mode: "rectangle"

                    MDTextField:
                        id: trans
                        hint_text: "Trans (0:Man, 1:Auto)"
                        mode: "rectangle"

                    MDFillRoundFlatButton:
                        text: "PREDICT PRICE"
                        size_hint_x: 0.9
                        pos_hint: {"center_x": 0.5}
                        font_size: "18sp"
                        md_bg_color: 0.2, 0.6, 0.9, 1
                        on_release: app.calculate()

                    MDCard:
                        orientation: "vertical"
                        padding: "15dp"
                        radius: [30,]
                        md_bg_color: 0.05, 0.1, 0.2, 1
                        size_hint_y: None
                        height: "180dp"
                        elevation: 4

                        MDLabel:
                            text: "AI VALUATION RANGE"
                            halign: "center"
                            font_style: "Overline"
                            theme_text_color: "Custom"
                            text_color: 0, 0.8, 1, 1
                            bold: True

                        MDLabel:
                            id: result_label
                            text: "READY"
                            halign: "center"
                            font_style: "H4"
                            bold: True
                            theme_text_color: "Custom"
                            text_color: 1, 1, 1, 1
                        MDLabel:
                            id: unit_label
                            text: ""
                            halign: "center"
                            font_style: "Subtitle2"
                            theme_text_color: "Custom"
                            text_color: 0, 1, 0.5, 1

                        MDBoxLayout:
                            padding: [10, 10, 10, 0]
                            size_hint_y: None
                            height: "30dp"

                            MDProgressBar:
                                id: progress
                                value: 0
                                color: 0, 0.7, 1, 1

                        MDBoxLayout:
                            orientation: 'horizontal'
                            padding: [5, 0]

                            MDLabel:
                                id: min_val
                                text: ""
                                halign: "left"
                                font_style: "Caption"
                                theme_text_color: "Custom"
                                text_color: 0.7, 0.7, 0.7, 1

                            MDLabel:
                                text: "10M Margin Error"
                                halign: "center"
                                font_style: "Caption"
                                theme_text_color: "Custom"
                                text_color: 0, 0.5, 0.8, 1

                            MDLabel:
                                id: max_val
                                text: ""
                                halign: "right"
                                font_style: "Caption"
                                theme_text_color: "Custom"
                                text_color: 0.7, 0.7, 0.7, 1

        MDBoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: "100dp"
            spacing: "5dp"
            padding: [0, 10]

            MDFillRoundFlatIconButton:
                icon: "instagram"
                text: "Visit HawreStack.Ai"
                md_bg_color: 0.89, 0.25, 0.37, 1
                pos_hint: {"center_x": .5}
                size_hint_x: 0.8
                on_release: app.open_instagram()

            MDLabel:
                text: "Developed by Eng. Hawre Omarpour"
                halign: "center"
                font_style: "Caption"
                theme_text_color: "Secondary"
                bold: True
'''

class CarApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_string(KV)

    def open_instagram(self, *args):
        webbrowser.open(
            "https://www.instagram.com/hawrestack.ai"
        )

    def calculate(self):
        try:

            price_val = get_car_price_prediction(
                year=int(
                    self.root.ids.year.text
                ),

                present_price=float(
                    self.root.ids.price.text
                ),

                kms_driven=int(
                    self.root.ids.kms.text
                ),

                fuel_petrol=int(
                    self.root.ids.fuel.text
                ),

                seller_individual=int(
                    self.root.ids.seller.text
                ),

                transmission_manual=int(
                    self.root.ids.trans.text
                )
            )

            lower_bound = price_val - 10
            upper_bound = price_val + 10

            unit_val = price_val / 100

            self.root.ids.result_label.text = (
                f"{price_val:,.0f} Million"
            )

            self.root.ids.unit_label.text = (
                f"Unit: {unit_val:.1f}"
            )

            self.root.ids.min_val.text = (
                f"{lower_bound:,.0f}"
            )

            self.root.ids.max_val.text = (
                f"{upper_bound:,.0f}"
            )

            self.root.ids.progress.value = 100
        except Exception:

            self.root.ids.result_label.text = (
                "Input Error!"
            )

            self.root.ids.unit_label.text = ""

            self.root.ids.min_val.text = ""

            self.root.ids.max_val.text = ""

            self.root.ids.progress.value = 0


if __name__ == "__main__":
    CarApp().run()

