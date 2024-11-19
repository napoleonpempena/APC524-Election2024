import pandas as pd
import numpy as np

df = pd.read_csv("data/president_polls.csv")

df = df.drop(
    columns=[
        "question_id",
        "poll_id",
        "cycle",
        "pollster_id",
        "sponsors",
        "sponsor_ids",
        "display_name",
        "pollster_rating_id",
        "pollster_rating_name",
        "population",
        "population_full",
        "methodology",
        "office_type",
        "seat_number",
        "seat_name",
        "sponsor_candidate_id",
        "sponsor_candidate",
        "internal",
        "partisan",
        "tracking",
        "nationwide_batch",
        "created_at",
        "notes",
        "url",
        "url_article",
        "url_topline",
        "url_crosstab",
        "subpopulation",
        "numeric_grade",
        "pollscore",
        "transparency_score",
        "sponsor_candidate_party",
        "endorsed_candidate_id",
        "endorsed_candidate_name",
        "endorsed_candidate_party",
        "source",
        "race_id",
        "election_date",
        "stage",
        "ranked_choice_reallocated",
        "ranked_choice_round",
        "hypothetical",
        "candidate_id",
    ]
)

# Add column for total voters (pct/100 * sample size)
df["votes"] = np.round(df["sample_size"] * df["pct"] / 100)
print(df)

df.to_csv("data/president_polls_cleaned.csv", index=False)
