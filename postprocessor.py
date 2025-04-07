import pandas as pd

def postprocess(model_classification):
    """
    model_classification: raw model classification
    returns "background" | "method" | "result"
    """
    classification = model_classification.lower()
    """
        Redundant after bottom 3 lines 
        classification = classification.strip() 
        classification =   classification[:-1] if classification[-1] == "s" else classification
    """
    classification =  "method" if "method" in classification else classification
    classification =  "result" if "result" in classification else classification
    classification =  "background" if classification not in ["method","background","result"] else classification
    return classification