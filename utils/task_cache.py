# _TASK_CACHE: Dict[str, Dict[str, str]] = {}
_TASK_CACHE = {
    # === rel-f1 (Formula 1) ===
    "driver-top3": {
        "dataset": "rel-f1",
        "description": (
            "For each driver predict if they will qualify in the top‑3 "
            "for a race in the next 1 month."
        ),
        "metric": "AUROC"
    },
    "driver-dnf": {
        "dataset": "rel-f1",
        "description": (
            "For each driver predict if they will DNF (did not finish) "
            "a race in the next 1 month."
        ),
        "metric": "AUROC"
    },
    "driver-position": {
        "dataset": "rel-f1",
        "description": (
            "Predict the average finishing position of each driver in all "
            "races in the next 2 months."
        ),
        "metric": "MAE"
    },

    # === rel-trial (Clinical Trials) ===
    "study-outcome": {
        "dataset": "rel-trial",
        "description": (
            "Predict if the clinical trial will achieve its primary outcome "
            "(defined as p‑value < 0.05)."
        ),
        "metric": "AUROC"
    },
    "study-adverse": {
        "dataset": "rel-trial",
        "description": (
            "Predict the number of affected patients with severe adverse events or "
            "death for the trial."
        ),
        "metric": "MAE"
    },
    "site-success": {
        "dataset": "rel-trial",
        "description": (
            "Predict the success rate of a trial site in the next 1 year."
        ),
        "metric": "MAE"
    },
    "condition-sponsor-run": {
        "dataset": "rel-trial",
        "description": (
            "Predict whether the sponsor (pharma/hospital) will run clinical trials "
            "for the condition in the next year."
        ),
        "metric": "MAP"
    },
    "site-sponsor-run": {
        "dataset": "rel-trial",
        "description": (
            "Predict whether this sponsor (pharma/hospital) will have a trial in "
            "the facility in the next year."
        ),
        "metric": "MAP"
    },

    # === rel-event (Event recommendation) ===
    "user-repeat": {
        "dataset": "rel-event",
        "description": (
            "Predict whether a user will attend an event (respond yes or maybe) "
            "in the next 7 days, after attending an event in the last 14 days."
        ),
        "metric": "AUROC"
    },
    "user-ignore": {
        "dataset": "rel-event",
        "description": (
            "Predict whether a user will ignore more than 2 event invitations "
            "in the next 7 days."
        ),
        "metric": "AUROC"
    },
    "user-attendance": {
        "dataset": "rel-event",
        "description": (
            "Predict how many events each user will respond yes or maybe "
            "to in the next 7 days."
        ),
        "metric": "MAE"
    },

    # === rel-amazon (E-commerce) ===
    "user-churn": {
        "dataset": "rel-amazon",
        "description": (
            "For each user, predict 1 if the customer does not review any product in "
            "the next 3 months, and 0 otherwise."
        ),
        "metric": "AUROC"
    },
    "item-churn": {
        "dataset": "rel-amazon",
        "description": (
            "For each product, predict 1 if the product does not receive any reviews in "
            "the next 3 months."
        ),
        "metric": "AUROC"
    },
    "user-ltv": {
        "dataset": "rel-amazon",
        "description": (
            "For each user, predict the $ value of the total number of products they buy "
            "and review in the next 3 months."
        ),
        "metric": "MAE"
    },
    "item-ltv": {
        "dataset": "rel-amazon",
        "description": (
            "For each product, predict the $ value of the total number of purchases and "
            "reviews it receives in the next 3 months."
        ),
        "metric": "MAE"
    },
    "user-item-purchase": {
        "dataset": "rel-amazon",
        "description": (
            "Predict the list of distinct items each customer will purchase in the next 3 months."
        ),
        "metric": "MAP"
    },
    "user-item-rate": {
        "dataset": "rel-amazon",
        "description": (
            "Predict the list of distinct items each customer will purchase and give a 5‑star review "
            "in the next 3 months."
        ),
        "metric": "MAP"
    },
    "user-item-review": {
        "dataset": "rel-amazon",
        "description": (
            "Predict the list of distinct items each customer will purchase and give a detailed review "
            "in the next 3 months."
        ),
        "metric": "MAP"
    },

    # === rel-stack (Stack‑Exchange) ===
    "user-engagement": {
        "dataset": "rel-stack",
        "description": (
            "For each user, predict if they will make any votes, posts, or comments in the next 3 months."
        ),
        "metric": "AUROC"
    },
    "user-badge": {
        "dataset": "rel-stack",
        "description": (
            "For each user, predict if they will receive a new badge in the next 3 months."
        ),
        "metric": "AUROC"
    },
    "post-votes": {
        "dataset": "rel-stack",
        "description": (
            "For each post, predict how many votes it will receive in the next 3 months."
        ),
        "metric": "MAE"
    },
    "user-post-comment": {
        "dataset": "rel-stack",
        "description": (
            "Predict a list of existing posts that a user will comment on in the next 2 years."
        ),
        "metric": "MAP"
    },
    "user-post-related": {
        "dataset": "rel-stack",
        "description": (
            "Predict a list of existing posts that users will link a given post to in the next 2 years."
        ),
        "metric": "MAP"
    },

    # === rel-hm (H&M) ===
    "user-churn": {
        "dataset": "rel-hm",
        "description": (
            "Predict the churn for a customer (no transactions) in the next week."
        ),
        "metric": "AUROC"
    },
    "item-sales": {
        "dataset": "rel-hm",
        "description": (
            "Predict the total sales for an article (the sum of prices of the associated transactions) in the next week."
        ),
        "metric": "MAE"
    },

    # === rel-avito (Avito ads) ===
    "user-visits": {
        "dataset": "rel-avito",
        "description": (
            "Predict whether each customer will visit more than one ad in the next 4 days."
        ),
        "metric": "AUROC"
    },
    "user-clicks": {
        "dataset": "rel-avito",
        "description": (
            "Predict whether each customer will click on more than one ad in the next 4 days."
        ),
        "metric": "AUROC"
    },
    "ad-ctr": {
        "dataset": "rel-avito",
        "description": (
            "Assuming the ad will be clicked in the next 4 days, predict the click-through-rate (CTR) for each ad."
        ),
        "metric": "MAE"
    },
    "user-ad-visit": {
        "dataset": "rel-avito",
        "description": (
            "Predict the list of ads a user will visit in the next 4 days."
        ),
        "metric": "MAP"
    },
}

def get_task_description(task_name: str) -> str:
    return _TASK_CACHE.get(task_name, {}).get("description", "")

def get_task_metric(task_name: str) -> str:
    return _TASK_CACHE.get(task_name, {}).get("metric", "")

def get_task_dataset(task_name: str) -> str:
    return _TASK_CACHE.get(task_name, {}).get("dataset", "")