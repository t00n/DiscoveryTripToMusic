import json

SETTINGS_FILE="settings.json"

def load_settings():
    return json.load(open(SETTINGS_FILE))

def save_settings(data):
    json.dump(data, open(SETTINGS_FILE, 'w'))

def save_best_features(best_features):
    data = load_settings()
    data['best_features'] = best_features
    save_settings(data)

data = load_settings()
best_features = data['best_features']