import re

print("Updating index.html...")
with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Replace $ logo with fingerprint SVG
old_svg = '<path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>'
new_svg = '''<path d="M12 11c0 2.5-1.5 4-3 4-1 0-1.5-.5-2-1.5"/>
                        <path d="M12 7c-2 0-4 1.5-4 4 0 3 1.5 5 4 5 1.5 0 3-1 3-3"/>
                        <path d="M12 3C7 3 4 6.5 4 11c0 4.5 2 7.5 5 9.5"/>
                        <path d="M16.5 8.5C18 10 19 12 19 14.5c0 3-2 5.5-5 7"/>
                        <path d="M12 15c1.5 0 2.5-1 2.5-2.5 0-1.5-1-2.5-2.5-2.5S9.5 11 9.5 12.5"/>'''
html = html.replace(old_svg, new_svg)

# Move disclaimer-banner inside app-header so it doesn't get cut off when sticky
html = html.replace('</header>\n\n    <!-- Medical Disclaimer Banner -->', '<!-- Medical Disclaimer Banner (Moved inside header) -->')
html = html.replace('        </div>\n    </section>\n\n\n\n    <!-- Main Workspace -->', '        </div>\n    </section>\n    </header>\n\n\n\n    <!-- Main Workspace -->')

# Add IDs to 70% texts so JS can update them
html = html.replace('the 70.0% reliability threshold.', 'the <span id="ui-thresh-1">70.0</span>% reliability threshold.')
html = html.replace('advisory threshold of 70%.', 'advisory threshold of <span id="ui-thresh-2">70</span>%.')
html = html.replace('passed (&tau; &ge; 70.0%).', 'passed (&tau; &ge; <span id="ui-thresh-3">70.0</span>%).')

# Add IDs to 112x112
html = html.replace('normalizes to 112×112', 'normalizes to <span id="ui-res-1">112</span>×<span id="ui-res-2">112</span>')

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Updating app.js...")
with open("static/js/app.js", "r", encoding="utf-8") as f:
    js = f.read()

# Add config fetching to app.js
init_code = """
// Globals
let CONFIG = { threshold: 70.0, image_size: 112, class_order: ["A+", "AB+", "B+", "O+"] };

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        if (res.ok) {
            CONFIG = await res.json();
            document.getElementById("ui-thresh-1").innerText = CONFIG.threshold.toFixed(1);
            document.getElementById("ui-thresh-2").innerText = CONFIG.threshold.toFixed(0);
            document.getElementById("ui-thresh-3").innerText = CONFIG.threshold.toFixed(1);
            document.getElementById("ui-res-1").innerText = CONFIG.image_size;
            document.getElementById("ui-res-2").innerText = CONFIG.image_size;
        }
    } catch (e) {
        console.warn("Could not fetch config, using defaults", e);
    }
}
fetchConfig();

"""
if "let CONFIG =" not in js:
    js = init_code + js

# Task 8: When the result is "Not Reliable", do not highlight the top class in the probability grid.
# In updateProbabilities, it sets the row color.
# We need to conditionally set it only if not needs_confirmation.
js = js.replace('row.style.background = "#eff6ff";', 'if (!needs_confirmation) { row.style.background = "#eff6ff"; }')
js = js.replace('row.style.borderLeft = "4px solid #3b82f6";', 'if (!needs_confirmation) { row.style.borderLeft = "4px solid #3b82f6"; }')
js = js.replace('row.style.fontWeight = "700";', 'if (!needs_confirmation) { row.style.fontWeight = "700"; }')

with open("static/js/app.js", "w", encoding="utf-8") as f:
    f.write(js)

print("Task 8 UI fixes applied.")
