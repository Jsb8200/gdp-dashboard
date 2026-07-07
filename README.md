# :earth_americas: GDP dashboard template

A simple Streamlit app showing the GDP of different countries in the world.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

---

## Magnitude Detector (LightGBM)

This repo also contains a LightGBM model that detects, before the move
starts, bars where a high-confluence / high-magnitude move is imminent.
See [MAGNITUDE_DETECTOR.md](MAGNITUDE_DETECTOR.md) for full docs.

```
# demo on synthetic data
python -m magnitude_detector.train --synthetic

# your data (see data/README.md for the CSV format)
python -m magnitude_detector.train --csv data/nq_5m.csv --out models/nq
python -m magnitude_detector.predict --model models/nq --csv data/nq_5m.csv
```
