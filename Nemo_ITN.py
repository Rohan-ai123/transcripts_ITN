pip install nemo-text-processing
pip install --upgrade nemo-toolkit
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

# Initialize the text normalizer for English
normalizer = Normalizer(lang="en", input_case="cased")

# Initialize the inverse text normalizer for English
inverse_normalizer = InverseNormalizer(lang="en")


input_text = """  """


normalized_text = normalizer.normalize(input_text, verbose=False)
print("Normalized Text:")
print(normalized_text)

written_text = inverse_normalizer.normalize(normalized_text, verbose=False)
print("Inverse Normalized Text:")
print(written_text)
