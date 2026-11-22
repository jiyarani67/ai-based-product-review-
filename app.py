
import csv
import io
import uuid

from flask import Flask, jsonify, render_template, request

from app.database import fetch_dashboard_stats, init_db, save_analysis
from app.ml_engine import ReviewMLPipeline
from app.text_utils import extract_keywords


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

pipeline = ReviewMLPipeline()
init_db()
pipeline.train()


def _collect_reviews_from_csv(file_storage) -> list[str]:
    if not file_storage:
        return []

    try:
        file_bytes = file_storage.read()
        decoded = file_bytes.decode("utf-8-sig", errors="ignore")
        reader = csv.DictReader(io.StringIO(decoded))

        if not reader.fieldnames:
            return []

        # Handle column names with extra spaces or different capitalization
        column_names = {
            name.strip().lower(): name for name in reader.fieldnames if name
        }

        review_column = column_names.get("review") or column_names.get("text")

        if not review_column:
            return []

        reviews = []

        for row in reader:
            review = (row.get(review_column) or "").strip()
            if review:
                reviews.append(review)

        return reviews

    except (csv.Error, UnicodeError, AttributeError, TypeError):
        return []


def _summarize_batch(results: list[dict]) -> dict:
    positive_reviews = [
        item["review"] for item in results
        if item["sentiment"] == "Positive"
    ]

    negative_reviews = [
        item["review"] for item in results
        if item["sentiment"] == "Negative"
    ]

    sentiment_counts = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0
    }

    for item in results:
        sentiment = item.get("sentiment")

        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

    recommendation = "Hold"

    if sentiment_counts["Positive"] > sentiment_counts["Negative"]:
        recommendation = "Recommended"

    elif sentiment_counts["Negative"] > sentiment_counts["Positive"]:
        recommendation = "Not Recommended"

    return {
        "total_reviews": len(results),
        "sentiment_distribution": sentiment_counts,
        "pros": extract_keywords(positive_reviews, top_n=8),
        "cons": extract_keywords(negative_reviews, top_n=8),
        "recommendation": recommendation,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze-review", methods=["POST"])
def analyze_single_review():
    payload = request.get_json(silent=True) or {}

    review_text = (payload.get("review_text") or "").strip()
    model = payload.get("model", "logistic")

    if not review_text:
        return jsonify({"error": "Review text is required."}), 400

    try:
        result = pipeline.analyze(review_text, model=model)

        save_analysis(
            "Single Input",
            f"single-{uuid.uuid4().hex[:8]}",
            result["review"],
            result["sentiment"],
            result["predicted_rating"]
        )

        return jsonify(result)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Single review analysis failed")
        return jsonify({"error": "Unable to analyze review. Please try again."}), 500


@app.route("/api/upload-reviews", methods=["POST"])
def upload_reviews():
    file = request.files.get("csv_file")
    product_name = (request.form.get("product_name") or "Uploaded Product").strip()
    model = request.form.get("model", "logistic")

    reviews = _collect_reviews_from_csv(file)

    if not reviews:
        return jsonify({
            "error": "CSV must contain a 'review' or 'text' column with data."
        }), 400

    batch_id = f"batch-{uuid.uuid4().hex[:8]}"
    results = []

    try:
        for review in reviews:
            analyzed = pipeline.analyze(review, model=model)
            results.append(analyzed)

            save_analysis(
                product_name,
                batch_id,
                analyzed["review"],
                analyzed["sentiment"],
                analyzed["predicted_rating"]
            )

        summary = _summarize_batch(results)

        return jsonify({
            "product_name": product_name,
            "batch_id": batch_id,
            "results": results,
            "summary": summary,
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("CSV review analysis failed")
        return jsonify({
            "error": "Unable to process uploaded reviews. Please try again."
        }), 500


@app.route("/api/compare-products", methods=["POST"])
def compare_products():
    file_one = request.files.get("product_a_file")
    file_two = request.files.get("product_b_file")

    product_a_name = (request.form.get("product_a_name") or "Product A").strip()
    product_b_name = (request.form.get("product_b_name") or "Product B").strip()

    model = request.form.get("model", "logistic")

    reviews_a = _collect_reviews_from_csv(file_one)
    reviews_b = _collect_reviews_from_csv(file_two)

    if not reviews_a or not reviews_b:
        return jsonify({
            "error": "Both files must be valid CSVs with review or text column."
        }), 400

    try:
        results_a = [
            pipeline.analyze(review, model=model) for review in reviews_a
        ]

        results_b = [
            pipeline.analyze(review, model=model) for review in reviews_b
        ]

        summary_a = _summarize_batch(results_a)
        summary_b = _summarize_batch(results_b)

        winner = "Tie"

        positive_a = summary_a["sentiment_distribution"]["Positive"]
        positive_b = summary_b["sentiment_distribution"]["Positive"]

        if positive_a > positive_b:
            winner = product_a_name

        elif positive_b > positive_a:
            winner = product_b_name

        return jsonify({
            "product_a": {
                "name": product_a_name,
                "summary": summary_a
            },
            "product_b": {
                "name": product_b_name,
                "summary": summary_b
            },
            "recommendation": winner,
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Product comparison failed")
        return jsonify({
            "error": "Unable to compare products. Please try again."
        }), 500


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    try:
        return jsonify(fetch_dashboard_stats())

    except Exception:
        app.logger.exception("Dashboard loading failed")
        return jsonify({"error": "Unable to load dashboard statistics."}), 500


@app.route("/api/evaluation", methods=["GET"])
def evaluation():
    try:
        return jsonify(pipeline.evaluation())

    except Exception:
        app.logger.exception("Model evaluation failed")
        return jsonify({"error": "Unable to load model evaluation."}), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "File is too large. Maximum upload size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
