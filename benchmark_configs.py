# only for bi_multiC
configs = {
    "nudity": {
        "data": {
            "forgot": "Nudity/forgot",
            "remove_word": "Nudity/remove_word"
        },
        "id2label": {0: 'nude', 1: 'safe'},
        "head_path": "Weights/bi_multiC/nudity",
    },
    "style_vangogh": {
        "data": {
            "forgot": "VanGogh/forgot",
            "remove_word": "VanGogh/remove_word"
        },
        "id2label": {0: 'non_vangogh', 1: 'vangogh'},
        "head_path": "Weights/bi_multiC/style_vangogh",
    },
    "object_church": {
        "data": {
            "forgot": "Church/forgot",
            "unrelated": "Church/unrelated"
        },
        "id2label": {0: 'other', 1: 'church'},
        "head_path": "Weights/bi_multiC/object_church",
        # "categories": ["Airplane", "Bird", "Cat", "Chair", "Parachute", "Motorcycle"]
    },
    "object_parachute": {
        "data": {
            "forgot": "Parachute/forgot",
            "unrelated": "Parachute/unrelated"
        },
        "id2label": {0: 'parachute', 1: 'other'},
        "head_path": "Weights/bi_multiC/object_parachute",
        # "categories": ["Airplane", "Bird", "Cat", "Chair", "Church", "Motorcycle"]

    },
}

