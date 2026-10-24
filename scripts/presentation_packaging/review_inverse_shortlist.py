from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path("E:/LumiCycleCache/experiments/night_day_demo_screen_20261024")
indices = [299, 269, 351, 411, 253, 187, 475, 109, 137, 300,
           263, 480, 130, 193, 97, 45, 288, 229, 329, 190]
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
for start in range(0, len(indices), 4):
    canvas = Image.new("RGB", (2048, 640), "#15171b")
    draw = ImageDraw.Draw(canvas)
    for cell, index in enumerate(indices[start:start+4]):
        x, y = cell % 2 * 1024, cell // 2 * 320
        draw.text((x+5, y+3), f"VAL {index:04d} | night input → unchanged model output", font=font, fill="white")
        for column, (folder, ext) in enumerate((("inputs", "jpg"), ("outputs", "png"))):
            with Image.open(root / folder / f"val_{index:04d}.{ext}") as image:
                assert image.size == (512, 288)
                canvas.paste(image.convert("RGB"), (x+column*512, y+32))
    canvas.save(root / "review" / f"native_shortlist_{start:02d}.png")
print("Native review sheets created")
