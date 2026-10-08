"""Run the detector on a sample image. Crops go to output/."""

from Characters_Detection import Characters_Detection

detector = Characters_Detection("Images/hori.jpg", R2L=False)
saved = detector.getCharacters("output/")
print(f"Saved {len(saved)} signs to output/")
