# DataSphere smoke dataset

The current bucket uses a folder-per-class layout under `classification_crops_padded/`.
The smoke job selects the six most populated valid classes with at least five images,
excludes non-taxonomic folders, and creates a deterministic 60/20/20 train/validation/test split.
This split is only for infrastructure and ClearML logging validation, not for reporting model quality.
