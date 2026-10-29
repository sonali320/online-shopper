document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const placeholderBox = document.getElementById("placeholder-box");
    const outputBox = document.getElementById("output-box");
    const btnPredict = document.getElementById("btn-predict-now");

    // Output elements
    const badgeVerdict = document.getElementById("badge-verdict");
    const tagEngineUsed = document.getElementById("tag-engine-used");
    const textScore = document.getElementById("text-score");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const verdictHeading = document.getElementById("verdict-heading");
    const verdictDescription = document.getElementById("verdict-description");
    const recommendationText = document.getElementById("recommendation-text");

    // Presets
    const presetHighIntent = document.getElementById("preset-high-intent");
    const presetCasual = document.getElementById("preset-casual");
    const presetBounce = document.getElementById("preset-bounce");

    const populateForm = (data) => {
        document.getElementById("input-page-values").value = data.pageValues;
        document.getElementById("input-exit-rates").value = data.exitRates;
        document.getElementById("input-bounce-rates").value = data.bounceRates;
        document.getElementById("input-product-related").value = data.productRelated;
        document.getElementById("input-product-duration").value = data.productDuration;
        document.getElementById("input-admin").value = data.admin;
        document.getElementById("input-admin-dur").value = data.adminDur;
        document.getElementById("input-info").value = data.info;
        document.getElementById("input-info-dur").value = data.infoDur;
        document.getElementById("input-month").value = data.month;
        document.getElementById("input-visitor-type").value = data.visitorType;
        document.getElementById("input-weekend").value = data.weekend;
    };

    if (presetHighIntent) {
        presetHighIntent.addEventListener("click", () => {
            populateForm({
                pageValues: 58.5,
                exitRates: 0.012,
                bounceRates: 0.002,
                productRelated: 36,
                productDuration: 1650.0,
                admin: 4,
                adminDur: 120.0,
                info: 2,
                infoDur: 45.0,
                month: "Nov",
                visitorType: "Returning_Visitor",
                weekend: "false"
            });
            form.requestSubmit();
        });
    }

    if (presetCasual) {
        presetCasual.addEventListener("click", () => {
            populateForm({
                pageValues: 0.0,
                exitRates: 0.065,
                bounceRates: 0.035,
                productRelated: 7,
                productDuration: 145.0,
                admin: 0,
                adminDur: 0.0,
                info: 0,
                infoDur: 0.0,
                month: "May",
                visitorType: "Returning_Visitor",
                weekend: "true"
            });
            form.requestSubmit();
        });
    }

    if (presetBounce) {
        presetBounce.addEventListener("click", () => {
            populateForm({
                pageValues: 0.0,
                exitRates: 0.20,
                bounceRates: 0.20,
                productRelated: 1,
                productDuration: 0.0,
                admin: 0,
                adminDur: 0.0,
                info: 0,
                infoDur: 0.0,
                month: "Feb",
                visitorType: "New_Visitor",
                weekend: "false"
            });
            form.requestSubmit();
        });
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const modelName = document.getElementById("select-model").value;
        const payload = {
            model_name: modelName,
            features: {
                PageValues: parseFloat(document.getElementById("input-page-values").value),
                ExitRates: parseFloat(document.getElementById("input-exit-rates").value),
                BounceRates: parseFloat(document.getElementById("input-bounce-rates").value),
                ProductRelated: parseInt(document.getElementById("input-product-related").value, 10),
                ProductRelated_Duration: parseFloat(document.getElementById("input-product-duration").value),
                Administrative: parseInt(document.getElementById("input-admin").value, 10),
                Administrative_Duration: parseFloat(document.getElementById("input-admin-dur").value),
                Informational: parseInt(document.getElementById("input-info").value, 10),
                Informational_Duration: parseFloat(document.getElementById("input-info-dur").value),
                Month: document.getElementById("input-month").value,
                VisitorType: document.getElementById("input-visitor-type").value,
                Weekend: document.getElementById("input-weekend").value === "true",
                SpecialDay: parseFloat(document.getElementById("input-special-day").value || "0.0"),
                OperatingSystems: parseInt(document.getElementById("input-os").value || "2", 10),
                Browser: parseInt(document.getElementById("input-browser").value || "2", 10),
                Region: parseInt(document.getElementById("input-region").value || "1", 10),
                TrafficType: parseInt(document.getElementById("input-traffic").value || "2", 10)
            }
        };

        btnPredict.disabled = true;
        btnPredict.innerHTML = "Computing Inference...";

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) {
                throw new Error(`Server returned HTTP ${resp.status}`);
            }

            const data = await resp.json();
            if (data.status !== "success") {
                throw new Error(data.message || "Unknown inference error");
            }

            const res = data.result;
            const isPurchase = res.purchase_prediction;
            const probPct = (res.purchase_probability * 100.0).toFixed(1);

            // Display results
            placeholderBox.classList.add("hidden");
            outputBox.classList.remove("hidden");

            tagEngineUsed.textContent = res.model_name;
            textScore.textContent = `${probPct}%`;
            progressBarFill.style.width = `${probPct}%`;

            if (isPurchase) {
                badgeVerdict.className = "verdict-badge badge-success";
                badgeVerdict.textContent = "PURCHASE LIKELY";
                progressBarFill.style.background = "linear-gradient(90deg, #6366f1, #10b981)";
                verdictHeading.textContent = "High Purchase Propensity Detected";
                verdictDescription.textContent = 
                    `The visitor has strong purchase signals (PageValues: ${payload.features.PageValues}, ` +
                    `Duration: ${payload.features.ProductRelated_Duration}s). Conversion probability is ${probPct}%.`;
                recommendationText.textContent = "Trigger high-priority checkout flow, limited-time discount, or free expedited shipping offer.";
            } else {
                badgeVerdict.className = "verdict-badge badge-danger";
                badgeVerdict.textContent = "NO PURCHASE EXPECTED";
                progressBarFill.style.background = "linear-gradient(90deg, #f59e0b, #ef4444)";
                verdictHeading.textContent = "Low Purchase Propensity";
                verdictDescription.textContent = 
                    `The visitor exhibits low engagement with zero or near-zero page values and elevated bounce/exit metrics. Purchase probability is only ${probPct}%.`;
                recommendationText.textContent = "Trigger engagement modal, popular product recommendations, or newsletter subscription incentives.";
            }
        } catch (err) {
            alert(`Inference failed: ${err.message}`);
        } finally {
            btnPredict.disabled = false;
            btnPredict.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg> Run Intention Prediction`;
        }
    });
});
