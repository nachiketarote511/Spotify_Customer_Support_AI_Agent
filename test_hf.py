import sys
sys.path.insert(0, 'k:/Hiver_assignment/Customer_Support_AI_Agent/backend')
from app.utils.hf_download import download_models_if_missing

download_models_if_missing('nachiketarote511/Spotify_Customer_Support_AI_Agent', 'k:/Hiver_assignment/Customer_Support_AI_Agent')
