configs = {
    "nudity": {
        "data": {
            "forgot": "Nudity/forgot",
            "remove_word": "Nudity/remove_word"
        },
        "id2label": {0: 'safe', 1: 'nude'},
        "head_path": "Weights/nudity",
    },
    "style_vangogh": {
        "data": {
            "forgot": "VanGogh/forgot",
            "remove_word": "VanGogh/remove_word"
        },
        "id2label": {0: 'non_vangogh', 1: 'vangogh'},
        "head_path": "Weights/style_vangogh",
    },
    "object_church": {
        "data": {
            "forgot": "Church/forgot",
            "unrelated": "Church/unrelated"
        },
        "id2label": {0: 'other', 1: 'church'},
        "head_path": "Weights/object_church",
        "categories": ["Airplane", "Bird", "Cat", "Chair", "Parachute", "Motorcycle"]
    },
    "object_parachute": {
        "data": {
            "forgot": "Parachute/forgot",
            "unrelated": "Parachute/unrelated"
        },
        "id2label": {0: 'other', 1: 'parachute'},
        "head_path": "Weights/object_parachute",
        "categories": ["Airplane", "Bird", "Cat", "Chair", "Church", "Motorcycle"]

    },
}
