from tkinter.tix import Tree
import uuid
from peewee import *
import datetime


db = SqliteDatabase('logs.sqlite3')

class BaseModel(Model):
    class Meta:
        database = db

class ConsentFormLog(BaseModel):
    uuid = CharField(unique=True)
    name = CharField()
    date = DateTimeField(default=datetime.datetime.now)
    is_accepted = BooleanField()

class RegistrationForm(BaseModel):
    uuid = CharField()
    user_data = CharField()

class PreSurveyLogs(BaseModel):
    uuid = CharField()
    user_data = CharField()

class PostNegoSurveyLogs(BaseModel):
    uuid = CharField()
    user_data = CharField()

class QuestionnaireLogs(BaseModel):
    uuid = CharField()
    user_data = CharField()

class IngredientSpecificationLogs(BaseModel):
    uuid = CharField()
    user_data = CharField(null=True)

class CuisineSpecificationLogs(BaseModel):
    uuid = CharField()
    user_data = CharField(null=True)

class OfferRoundLogs(BaseModel):
    uuid = CharField()
    interaction = BooleanField()
    health_score = FloatField()
    overall_score = FloatField()
    offer_number = IntegerField()
    recipe_id = CharField()
    feedback = CharField(null=True)
    explanation = CharField(null=True)

class RoundSummary(BaseModel):
    uuid = CharField()
    total_offers = IntegerField()
    is_terminated = BooleanField()

if __name__ == "__main__":
    db.create_tables([CuisineSpecificationLogs, IngredientSpecificationLogs, ConsentFormLog, RegistrationForm, PreSurveyLogs, PostNegoSurveyLogs, QuestionnaireLogs, OfferRoundLogs, RoundSummary])