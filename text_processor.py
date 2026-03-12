import re
from attributes import attributes

class TextProcessor:
    @staticmethod
    def preprocess_caption(caption):
        found_attributes = {category: [] for category in attributes.keys()}
        
        for category, attrs in attributes.items():
            for attr in attrs:
                type_match = re.search(rf'\b{attr["type"]}\b', caption, re.IGNORECASE)
                if type_match:
                    color_match = re.search(rf'\b(?P<color>\w+)\s+{attr["type"]}\b', 
                                          caption, re.IGNORECASE)
                    color = color_match.group("color") if color_match else None
                    found_attributes[category].append({
                        "type": attr["type"], 
                        "color": color
                    })
                    
        return found_attributes