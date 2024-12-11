import plotly.express as px
from jinja2 import Template

import pandas as pd

# Relative imports
import pie_chart
import state_data
import date_data

def output_to_html(chart_information: dict):

    # Paths for the HTML template and the output HTML page
    template_path = "/home/gr7610/APC524-Election2024/src/apc524-election2024/jinja_template.html"
    output_path = "/home/gr7610/APC524-Election2024/src/index.html"

    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render(chart_information))

# Set parameters
state = "Texas"
date = "2024-11-04"

# Begin data loading
data = pd.read_csv("/home/gr7610/APC524-Election2024/data/president_polls_cleaned.csv")

# Process the data
state_data, combined_state_data = state_data.get_state_data(state, data)
date_data, combined_date_data = date_data.get_date_data(date, data)

# Generate chart objects
state_pie_chart = pie_chart.pie_chart(combined_state_data, state=state)
date_pie_chart = pie_chart.pie_chart(combined_date_data, date=date)

# Generate Jinja objects
jinja_data = {"date_fig": date_pie_chart.to_html(full_html=False, 
                                                include_plotlyjs='cdn'),
              "state_fig": state_pie_chart.to_html(full_html=False, 
                                                include_plotlyjs='cdn')}

# Output data to HTML
output_to_html(chart_information = jinja_data)
