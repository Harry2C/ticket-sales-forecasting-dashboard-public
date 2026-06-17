# Model Artifacts

This directory is reserved for trained model artifacts once the platform moves beyond the in-app baseline forecaster.

The current application trains a replaceable random-forest forecasting model at runtime from the loaded sample or production data. Persisted artifacts should be written under `models/artifacts/`, which is ignored by git.

