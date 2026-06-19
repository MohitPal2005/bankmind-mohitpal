# Architectural Exploration & Validation Insights

### Global Analytical Questions (All Tracks)

#### 1. What percentage of customers in your dataset have y = yes? What does this imbalance mean for how you'd evaluate a model?
About 11.7% of customers in the dataset actually subscribed. Because "no" is so much more common than "yes", looking at global accuracy is misleading—a model could just guess "no" every time and be 88% accurate. Instead, we must evaluate using F1-score, Precision, and Recall to see how well it catches the rare positive cases.

#### 2. Which job category had the highest subscription rate? Does this make sense to you intuitively?
Students and retired individuals had the highest subscription rates. This makes logical sense because these groups usually seek safe, low-risk, fixed-return savings options like term deposits, rather than high-risk investments.

### Track B Engine Framework Queries

#### 3. Which feature had the highest importance in your tree-based model? Why do you think that is?
The `duration` feature (how long the last phone call lasted) was the most important at ~34.4% importance. Longer call durations usually indicate that customers were genuinely interested and asking questions about the banking product, so the model heavily relies on this to predict success.

#### 4. Why is F1 a better metric than accuracy for this particular dataset?
Our Random Forest achieved an Accuracy of ~91%, but its F1-score is ~0.46. F1 is better here because it balances False Positives (wasting an RM's time calling a bad lead) and False Negatives (missing out on a sale), giving us a true picture of the model's business value on the imbalanced data.

#### 5. Pick one of your 5 sample predictions. Do you actually agree with the model's call, given that customer's features? Walk through your thinking.
Looking at Customer ID 45163 (Age: 71, Job: retired, Balance: €2064, Housing/Personal Loans: No). The model predicted they "Will Subscribe" with a 58.00% probability. I agree with this call because retired individuals with a healthy bank balance and zero active debt have the financial freedom and low-risk appetite suited for a simple term deposit.

### Track C Infrastructure Engineering Queries

#### 6. What would likely break first if 200 RMs were hitting your /predict endpoint simultaneously? What's one thing you'd change?
A single FastAPI process could become overloaded and time out if hundreds of users send requests simultaneously. To fix this, I would deploy multiple worker processes using Gunicorn (with Uvicorn workers) and add a load balancer to distribute the traffic.

#### 7. What does the LLM explanation actually add over just showing a probability score?
A raw percentage tells an RM *what* to expect, but the LLM tells them *why* and *how* to act. By converting SHAP math into a plain-English strategy, the RM instantly gets conversational talking points tailored to that specific customer's profile.