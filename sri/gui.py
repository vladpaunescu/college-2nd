import Tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
from summary import get_image, process_text


def onclick(text, panel):
    print text
    sentences = process_text(text)
    for s in sentences:
        print(s)
    print "best sentence ", sentences[0]
    data = get_image(sentences[0][1])
    img = ImageTk.PhotoImage(Image.open(BytesIO(data)))
    panel.configure(image = img)
    panel.image = img

root = tk.Tk()

text = tk.Text(root, font=('Helvetica', 16))
text.grid(row=0, column=0)

button = tk.Button(root, font=('Monospace', 16), command=lambda: onclick(text.get("1.0", tk.END), panel))
button.grid(row=1, column=0)
button["text"] = 'Find image'

# img = ImageTk.PhotoImage(Image.open('img.jpg'))
panel = tk.Label(root)
panel.grid(row=0, column=1)

root.bind("<Return>", lambda(e): onclick(text.get("1.0", tk.END), panel))
root.mainloop()
