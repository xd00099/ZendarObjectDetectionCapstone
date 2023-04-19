import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Automatic Radar Labeling Viewer')
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        # self.attributes('-fullscreen', True)

        self.front_folder = 'outputs/front_view'
        self.left_folder = 'outputs/left_view'
        self.right_folder = 'outputs/right_view'
        self.top_folder = 'outputs/bird_eye_view'
        self.image_index = 0
        self.playing = False
        self.rate = 50  # in milliseconds

        self.create_widgets()
        self.load_images()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        self.front_view = ttk.Label(self.main_frame)
        self.left_view = ttk.Label(self.main_frame)
        self.right_view = ttk.Label(self.main_frame)
        self.top_view = ttk.Label(self.main_frame)

        self.front_view.grid(row=0, column=1)
        self.left_view.grid(row=1, column=0)
        self.right_view.grid(row=1, column=2)
        self.top_view.grid(row=1, column=1)

        self.file_name_label = ttk.Label(self.main_frame, text="")
        self.file_name_label.grid(row=2, column=1, pady=10)

        self.prev_button = ttk.Button(self.main_frame, text='Previous', command=self.prev_image)
        self.next_button = ttk.Button(self.main_frame, text='Next', command=self.next_image)
        self.play_button = ttk.Button(self.main_frame, text='Play', command=self.play)
        self.beginning_button = ttk.Button(self.main_frame, text='Return to Start', command=self.back_to_begin)

        self.rate_label = ttk.Label(self.main_frame, text="Speed (ms):")
        self.rate_label.grid(row=4, column=0, padx=(0, 5), sticky='E')
        self.rate_combobox = ttk.Combobox(self.main_frame, values=(10, 50, 200, 500, 1000), state="readonly")
        self.rate_combobox.set(50)
        self.rate_combobox.grid(row=4, column=1, sticky='W')

        self.prev_button.grid(row=3, column=0, pady=10)
        self.beginning_button.grid(row=3, column=1, pady=10)
        self.next_button.grid(row=3, column=2, pady=10)
        self.play_button.grid(row=4, column=1, columnspan=2)

    def load_images(self):
        self.front_images = sorted(os.listdir(self.front_folder))
        self.left_images = sorted(os.listdir(self.left_folder))
        self.right_images = sorted(os.listdir(self.right_folder))
        self.top_images = sorted(os.listdir(self.top_folder))

        self.update_views()

    def update_views(self):
        self.update_view(self.front_view, os.path.join(self.front_folder, self.front_images[self.image_index]))
        self.update_view(self.left_view, os.path.join(self.left_folder, self.left_images[self.image_index]))
        self.update_view(self.right_view, os.path.join(self.right_folder, self.right_images[self.image_index]))
        self.update_view(self.top_view, os.path.join(self.top_folder, self.top_images[self.image_index]))

        self.file_name_label.config(text=self.front_images[self.image_index])

    def update_view(self, label, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        max_width = screen_width // 2.7
        max_height = screen_height // 2.7

        width_ratio = max_width / img.shape[1]
        height_ratio = max_height / img.shape[0]
        min_ratio = min(width_ratio, height_ratio)

        new_width = int(img.shape[1] * min_ratio)
        new_height = int(img.shape[0] * min_ratio)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        label.config(image=img)
        label.image = img

    def prev_image(self):
        self.image_index = (self.image_index - 1) % len(self.front_images)
        self.update_views()

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.front_images)
        self.update_views()

    def back_to_begin(self):
        self.image_index = 0
        self.update_views()

    def play(self):
        if not self.playing:
            self.playing = True
            self.play_button.config(text='Stop')
            self.rate = int(self.rate_combobox.get())
            self.next_frame()
        else:
            self.playing = False
            self.play_button.config(text='Play')

    def next_frame(self):
        if self.playing:
            self.next_image()
            self.after(self.rate, self.next_frame)

if __name__ == '__main__':
    app = ImageViewer()
    app.mainloop()
