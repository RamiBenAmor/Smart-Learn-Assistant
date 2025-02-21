from textblob import TextBlob
def Correction_spelling(s) :
  return TextBlob(s).correct()
