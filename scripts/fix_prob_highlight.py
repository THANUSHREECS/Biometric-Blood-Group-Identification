import os

with open("static/js/app.js", "r", encoding="utf-8") as f:
    js = f.read()

# Change isTop to depend on needs_confirmation
old_code = "const isTop = (bg === data.blood_group);"
new_code = "const isTop = (bg === data.blood_group) && !(data.needs_confirmation);"
if old_code in js:
    js = js.replace(old_code, new_code)
    with open("static/js/app.js", "w", encoding="utf-8") as f:
        f.write(js)
    print("Fixed JS highlight.")
else:
    print("Could not find old_code in JS.")
