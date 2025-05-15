from os.path import exists
import tkinter as tk
import numpy as np 
import keras

CELL_SIZE = 20
GRID_SIZE = 28
BRUSH_RADIUS = 1  # Radius of brush in cells (1 = 3x3 brush, 2 = 5x5, etc.)

class DrawingBoard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Guess Digit NN")
        self.canvas = tk.Canvas(self, width=CELL_SIZE*GRID_SIZE, height=CELL_SIZE*GRID_SIZE, bg='white')
        self.canvas.pack()
        
        self.rects = {}
        self.grid_data = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        self.draw_grid()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        
        self.bind("<space>", self.on_space_pressed)
        self.bind("<Button-3>", self.on_space_pressed)
        self.bind("<Double-Button-3>", self.clear)

        # BOTTOM FRAME

        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(fill = "x", padx = 5, pady= 5)

        # LABEL FRAME

        self.label_frame = tk.Frame(self.bottom_frame)
        self.label_frame.grid(row = 0, column = 0, sticky = "w")

        self.prediction_label_1 = tk.Label(self.label_frame, text = "Predictions will be shown here.")
        self.prediction_label_1.pack(anchor="w")

        self.prediction_label_2 = tk.Label(self.label_frame, text = "")
        self.prediction_label_2.pack(anchor="w")

        # end of LABEL FRAME

        self.clear_button = tk.Button(self.bottom_frame, text="Clear", command=self.clear)
        self.clear_button.grid(row=0, column = 1, sticky="e")

        self.bottom_frame.columnconfigure(0, weight=7)
        self.bottom_frame.columnconfigure(1, weight=3)

        # end of BOTTOM FRAME

        self.help_label = tk.Label(self, text = "Draw and Press Right-Mouse-Button or Space to see predictions.")
        self.help_label.pack(pady=10)

        self.nn_model = keras.models.load_model("./model/numbers_convnet.keras")

    def draw_grid(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='gray')
                self.rects[(row, col)] = rect

    def draw(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        for dr in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
            for dc in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
                r = row + dr
                c = col + dc
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    rect_id = self.rects[(r, c)]
                    self.canvas.itemconfig(rect_id, fill='black')
                    self.grid_data[r][c] = 1

    def clear(self, event=None):
        for (row, col), rect_id in self.rects.items():
            self.canvas.itemconfig(rect_id, fill='white')
        self.grid_data = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self.help_label.config(text = "Draw and Press Right-Mouse-Button or Space to see predictions.")
        self.prediction_label_1.config(text = "Predictions will be shown here.")
        self.prediction_label_2.config(text = "")


    def get_board(self):
        return self.grid_data

    def on_space_pressed(self, event):
        board = self.get_board()
        board_data = np.array([
            board
        ])
        board_data = np.expand_dims(board_data, axis=-1)
        predictions = self.nn_model.predict(board_data, verbose = 0).squeeze()

        predicted_numbers = np.argsort(predictions)[-2:][::-1]
        prediction_confidence = [predictions[i] for i in predicted_numbers]

        self.prediction_label_1.config(text = f"First Guess: {predicted_numbers[0]} ( {float(prediction_confidence[0]):.2f} )")
        self.prediction_label_2.config(text = f"Second Guess: {predicted_numbers[1]} ( {float(prediction_confidence[1]):.2f} )")

        self.help_label.config(text = "Double click Right-Mouse-Button to clear canvas.")




