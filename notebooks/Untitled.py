# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML

# Example DataFrame
df = pd.DataFrame({
    "A": range(5),
    "B": ["foo", "bar", "baz", "qux", "quux"]
})

# Convert DataFrame to HTML
df_html = df.to_html(classes="dataframe table table-striped")

# Create toggle button and output container
button = widgets.ToggleButton(value=False, description="Show/Hide Table")
output = widgets.Output()

def toggle_table(change):
    output.clear_output()
    if change["new"]:
        with output:
            display(HTML(df_html))

button.observe(toggle_table, names="value")

# Display the toggle and the output
display(button, output)
# -


