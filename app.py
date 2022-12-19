from asyncio import selector_events
import pickle
from urllib.parse import uses_relative
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import random
from anytree import Node
from lxml import etree as ET
from anytree.exporter import DictExporter
import uuid
from owlready2 import *
from pip import main
import numpy as np
from health_module import HealthModule
from user_rating_module import UserRatingModule
import ast
import time
from database_models import *

onto = get_ontology(
    "/home/kalimdor/Projects/expectation/expectation-backend/FoodMenu.owl").load()


def get_users_ontology_dict():
    users = list(onto.User.subclasses())
    user_to_onto = {}
    for user in users:
        user_to_onto[user.name.lower()] = user

    return user_to_onto


app = Flask(__name__)
recipes = pd.read_csv("./datasets/archive/core-data_recipe.csv")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def get_explanation(recipe, user_profile, user_cuisine_prefs, user_ingr_prefs):
    parsed_nutritions = ast.literal_eval(recipe.nutritions)
    user_bmr = user_profile["user_bmr"]
    calorie_coverage = parsed_nutritions["calories"]["amount"] / \
        user_bmr
    protein_coverage = parsed_nutritions["protein"]["amount"] / (user_profile["weight"] * 0.8)
    
    print(protein_coverage)

    protein_count = float(str(parsed_nutritions["protein"]["displayValue"]).replace(" ", "").replace("<", ""))
    vitamin_c_percentage = int(float(str(parsed_nutritions["vitaminC"]["percentDailyValue"]).replace(" ", "").replace("<", "")))
    iron_percentage = int(float(str(parsed_nutritions["iron"]["percentDailyValue"]).replace(" ", "").replace("<", "")))
    cholesterol_amount = float(str(parsed_nutritions["cholesterol"]["amount"]).replace(" ", "").replace("<", ""))


    internet_rating_score = int(recipe["internet_rating_score"])

    user_cuisine_match = recipe.Cuisine in user_cuisine_prefs
    user_ingrs_match = [
        value for value in recipe.ingredients if value in user_ingr_prefs]

    explanation_starters = [
        "This recipe was recommended to you given various healthiness considerations as well as your preferences towards food which you specificed."
        ]

    explanations = {
        "healthy": [f"You should eat this food because it covers a good portion of your necessary daily calorie intake by {calorie_coverage:.1%}. You should eat {user_bmr} for a balanced diet and maintain a healthy weight!",
                    f"This recipe contains {protein_count} grams of protein, which is about {protein_coverage:.1%} of your daily requirement. Your body needs proteins from your organs to your muscles, and consuming the necessary amount of it is important!"
                    f"You need to eat "],
        "liked": [f"This recipe was also rated {internet_rating_score} stars by the community and you might like it too!"]
    }

    if vitamin_c_percentage > 30:
        explanations["healthy"].append(
            f"Vitamin C is needed for the growth and repair of tissues in all parts of your body. This recipe contains: {vitamin_c_percentage}% of it, and eating it will make you feel more energetic!")
    
    if iron_percentage > 20:
        explanations["healthy"].append(
            f"Lack of Iron in your system could be critical, causing a disease that is known as iron deficiency anaemia. This recipe supplies you with {iron_percentage} of it!")
    
    if cholesterol_amount < 75:
        explanations["healthy"].append(
            f"This recipe has a very low cholesterol count ({cholesterol_amount}). Cholesterol is linked with a higher risk of cardiovascular disease, and this food is great for low amounts of it. ")

    if user_cuisine_match:
        explanations["liked"].append(
            f"This recipe is also a part of the cuisine you like: {recipe.Cuisine}")

    if user_ingrs_match:
        explanations["liked"].append(
            f"This recipe contains the food you wanted: {' and '.join(user_ingrs_match)}")

    return random.choice(explanation_starters) + " " + " ".join(random.sample(explanations["healthy"], 2)) + " " + random.choice(explanations["liked"])


def create_user_recipes(habits):
    if not habits:
        custom_recipes = pd.read_pickle(f"./datasets/habits/Vanilla_mod.pkl")
    else:
        custom_recipes = pd.read_pickle(f"./datasets/habits/{habits[0]}_mod.pkl")
        for habit in habits[1:]:
            new_base = pd.read_pickle(f"./datasets/habits/{habit}_mod.pkl")
            both = pd.merge(custom_recipes, new_base, how='inner', on=[
                            'recipe_id'], suffixes=["", "_y"])
            both = both.drop([x for x in both.columns if "_y" in x], axis=1)

    custom_recipes.drop_duplicates(["recipe_name"], inplace=True)

    return custom_recipes


def create_health_rating_scores(user_profile):
    nutritional_values = pd.read_pickle('./recipe_calories.pkl')

    user_based_score_df = pd.DataFrame()
    user_based_score_df.index = nutritional_values.index

    health_module = HealthModule(user_profile)
    health_module.calculate_scores(nutritional_values, user_based_score_df)
    user_rating_module = UserRatingModule()
    user_rating_module.store_users_rating(user_based_score_df)

    user_based_score_df.dropna(inplace=True)

    user_based_score_df['overall'] = (
        1 * user_based_score_df['amr_score']
        + 1 * user_based_score_df['health_score']
        + 1 * user_based_score_df['internet_rating_score'])/3
    user_based_score_df.sort_values(
        by=['overall'], inplace=True, ascending=False)

    return health_module, user_based_score_df[["amr_score", "health_score", "internet_rating_score", "overall"]]

def generate_custom_recipes(generated_uuid, user_profile, interaction):
    custom_recipes = create_user_recipes(user_profile["habits"])
    user_health_module, user_health_score = create_health_rating_scores(
        user_profile)

    merged_profile = custom_recipes.merge(
        user_health_score, on="recipe_id", how='left').dropna()
    
    merged_profile["overall"] = merged_profile.overall.transform(lambda row: row / row.max(), axis=0)
    
    merged_profile = merged_profile[
        ~merged_profile.ingredients_rep
        .map(lambda row: np.isin(row.split("^"), user_profile["allergies"]).any())]

    merged_profile.sort_values(by=["overall"], ascending=False).to_pickle(f"./cache/{interaction}/{generated_uuid}.pkl")

    return user_health_module.user_bmr

@app.route("/save-form", methods=["POST"])
def save_form():
    submitted_form = request.json
    generated_uuid = submitted_form.pop("uuid")
    form_type = submitted_form.pop("form_type")

    databases = {
        "registration": RegistrationForm,
        "pre_survey": PreSurveyLogs,
        "post_nego_survey": PostNegoSurveyLogs,
        "questionnaire": QuestionnaireLogs,
    }

    databases[form_type].create(uuid=generated_uuid, user_data=submitted_form)

    return jsonify({"status": True})


@app.route("/end-negotiation", methods=["POST"])
def end_negotiation():
    submitted_form = request.json
    generated_uuid = submitted_form.pop("sessionID")
    terminated = submitted_form.pop("terminated")
    interaction = "interactive" if submitted_form.pop("explanationSection") else "regular"

    if interaction:
        total_offers = len(list(OfferRoundLogs.select().where((OfferRoundLogs.uuid == generated_uuid), (OfferRoundLogs.explanation != ""))))
    else:
        total_offers = len(list(OfferRoundLogs.select().where((OfferRoundLogs.uuid == generated_uuid), (OfferRoundLogs.explanation == ""))))

    query_contents = {
        "uuid": generated_uuid,
        "total_offers": str(total_offers+1),
        "is_terminated": terminated,
    }

    RoundSummary.create(**query_contents)

    meal_type_swap = {"lunch": "dinner", "dinner": "lunch"}
    interaction_swap = {"interactive": "regular", "regular": "interactive"}

    with open(f'./cache/user_profiles/{generated_uuid}.pkl', 'rb') as handle:
        user_profile = pickle.load(handle)   

    user_profile["mealtype"] = meal_type_swap[user_profile["mealtype"]]
    user_bmr = generate_custom_recipes(generated_uuid, user_profile, interaction_swap[interaction])

    return jsonify({"status": True})


@app.route("/submit-cuisine-specifications", methods=["POST"])
def submit_cuisine():
    submitted_form = request.json
    generated_uuid = submitted_form.pop("sessionID")
    interaction = "interactive" if submitted_form.pop("explanationSection") else "regular"
    
    unwanted_cuisines = submitted_form["unwantedItems"]
    wanted_cuisines = submitted_form["wantedItems"]

    whitelist = {
        "Middle Eastern": ["Middle East", "Greece"],
        "Asian": ["Japan", "China", "Thailand"],
    }

    for item in wanted_cuisines:
        if item in whitelist.keys():
            wanted_cuisines.extend(whitelist[item])

    for item in unwanted_cuisines:
        if item in whitelist.keys():
            unwanted_cuisines.extend(whitelist[item])

    query_data = {
        "uuid": generated_uuid,
        "user_data": {"wanted_items": wanted_cuisines, "unwanted_items": unwanted_cuisines} 
    }
    
    CuisineSpecificationLogs(**query_data).save()

    df = pd.read_pickle(f"./cache/{interaction}/{generated_uuid}.pkl")
    df = df[~df['Cuisine'].isin(unwanted_cuisines)]
    df["cuisine_matching_score"] = df.Cuisine.apply(
        lambda row: 1 if row in wanted_cuisines else 0)
    df.to_pickle(f"./cache/{interaction}/{generated_uuid}.pkl")

    return jsonify({"status": True})


@app.route("/submit-ingredient-specifications", methods=["POST"])
def submit_ingredients():
    submitted_form = request.json
    generated_uuid = submitted_form.pop("sessionID")
    interaction = "interactive" if submitted_form.pop("explanationSection") else "regular"

    unwanted_ingrs = submitted_form["unwantedItems"]
    wanted_ingrs = submitted_form["wantedItems"]
    
    tmp = []
    for item in wanted_ingrs:
        sub_list = item.split(" AND ")
        tmp.extend(sub_list)

        for sub_item in sub_list:
            if isinstance(onto[sub_item], owlready2.entity.ThingClass):
                tmp.extend([x.name.replace("_", " ")
                                    for x in onto[sub_item].instances()])
    wanted_ingrs = tmp

    tmp = []
    for item in unwanted_ingrs:
        sub_list = item.replace("NOT ", "").replace("(", "").replace(")", "").split(" AND ")
        tmp.extend(sub_list)

        for sub_item in sub_list:
            if isinstance(onto[sub_item], owlready2.entity.ThingClass):
                tmp.extend([x.name.replace("_", " ")
                                      for x in onto[sub_item].instances()])
    
    unwanted_ingrs = tmp

    df = pd.read_pickle(f"./cache/{interaction}/{generated_uuid}.pkl")
    df = df[~df.matched_ingrs.map(lambda row: np.isin(row, unwanted_ingrs).any())]

    matches = df.matched_ingrs.apply(lambda row: len(np.intersect1d(row, wanted_ingrs)))
    df["final_matching_score"] = df["cuisine_matching_score"] + matches
    if df["final_matching_score"].max() != 0:
        df["final_matching_score"] = df["final_matching_score"].transform(lambda x: x / x.max(), axis=0)

    df["total_score"] = df["final_matching_score"] * 0.5 + df["overall"] * 0.5 
    df.to_pickle(f"./cache/{interaction}/{generated_uuid}.pkl")

    query_data = {
        "uuid": generated_uuid,
        "user_data": {"wanted_items": wanted_ingrs, "unwanted_items": unwanted_ingrs} 
    }

    IngredientSpecificationLogs(**query_data).save()

    return jsonify({"status": True})


@app.route("/save-consent", methods=["POST"])
def save_consent():
    submitted_form = request.json
    generated_uuid = str(uuid.uuid4())
    name = submitted_form.pop("name")

    ConsentFormLog.create(uuid=generated_uuid, name=name, is_accepted=True)
    explanation_start = len(list(ConsentFormLog.select())) % 2 == 0

    return jsonify({"status": True, "explanation_start": explanation_start, "uuid": generated_uuid})

@ app.route("/generate-user-profile", methods=["POST"])
def generate_user_profile():
    submitted_form = request.json
    generated_uuid = submitted_form["uuid"]
    interaction = "interactive" if submitted_form.pop("explanationSection") else "regular"

    user_profile = request.json
    user_profile["sports"] = submitted_form["sports"]
    
    user_profile["allergies"] = submitted_form["allergies"]

    user_bmr = generate_custom_recipes(generated_uuid, user_profile, interaction)

    user_profile["user_bmr"] = user_bmr
    with open(f'./cache/user_profiles/{generated_uuid}.pkl', 'wb') as handle:
        pickle.dump(user_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return jsonify({"status": True})

@ app.route("/get-recipe", methods=["POST"])
def get_recipe():
    submitted_form = request.json
    uuid = request.json["sessionID"]
    explanation_section = submitted_form.pop("explanationSection")
    interaction = "interactive" if explanation_section else "regular"

    feedback = request.json.get("feedback", None)
    unwanted_ingrs = [joined_item.strip() for item in feedback for joined_item in item.split("-")[1:]]

    user_recipes = pd.read_pickle(f"./cache/{interaction}/{uuid}.pkl")
    user_recipes = user_recipes[~user_recipes.ingredients_rep.map(lambda row: np.isin(row.split("^"), unwanted_ingrs).any())]

    user_recipes.sort_values("total_score", ascending=False, inplace=True)
    sample_recipe = user_recipes.iloc[0]

    user_recipes.drop(index=user_recipes.index[0],
                                axis=0,
                                inplace=True)  # fix later

    recipe_dict = sample_recipe[["image_url",
                                 "recipe_name", 
                                 "ingredients_rep", 
                                 "Cuisine",
                                 "cooking_directions"]].to_dict()

    recipe_dict["cooking_directions"] = ast.literal_eval(recipe_dict["cooking_directions"])['directions'].split("\n")
    
    cuisine_whitelist = {
        "Greece": "Middle East",
    }

    recipe_dict["ingredients_rep"] = recipe_dict["ingredients_rep"].split("^")
    recipe_dict["Cuisine"] = cuisine_whitelist.get(recipe_dict["Cuisine"], recipe_dict["Cuisine"])

    with open(f'./cache/user_profiles/{uuid}.pkl', 'rb') as handle:
        user_profile = pickle.load(handle)    

    user_prefs = ast.literal_eval(list(IngredientSpecificationLogs.select().where(IngredientSpecificationLogs.uuid == uuid))[-1].user_data)['wanted_items']
    cuisine_prefs = ast.literal_eval(list(CuisineSpecificationLogs.select().where(CuisineSpecificationLogs.uuid == uuid))[-1].user_data)['wanted_items']

    explanation = "" if not explanation_section else get_explanation( 
                                                            sample_recipe, 
                                                            user_profile,
                                                            cuisine_prefs,
                                                            user_prefs)

    if explanation_section:
        total_offers = len(list(OfferRoundLogs.select().where((OfferRoundLogs.uuid == uuid), (OfferRoundLogs.explanation != ""))))
    else:
        total_offers = len(list(OfferRoundLogs.select().where((OfferRoundLogs.uuid == uuid), (OfferRoundLogs.explanation == ""))))

    query_dict = {
        "uuid": uuid,
        "interaction": interaction, #whether its interactive
        "recipe_id": sample_recipe.recipe_id,
        "overall_score": sample_recipe.overall,
        "explanation": explanation,
        "health_score": sample_recipe.health_score,
        "offer_number": total_offers+1,
    }

    if feedback:
        query_dict["feedback"] = feedback

    OfferRoundLogs(**query_dict).save()
    user_recipes.to_pickle(f"./cache/{interaction}/{uuid}.pkl")

    return jsonify({"status": True, "explanation": explanation, "recipe": recipe_dict})


@ app.route("/get-recipe-ingredients", methods=["POST"])
def get_recipe_ingredients():
    parser = ET.XMLParser(remove_blank_text=True)
    test_tree = ET.parse('./FoodMenu.owl', parser)
    test_root = test_tree.getroot()
    uuid = request.json["uuid"]

    try:
        with open(f'./cache/user_profiles/{uuid}.pkl', 'rb') as handle:
            user_profile = pickle.load(handle)
    except FileNotFoundError as err:
        user_profile = None

    user_habits_does_not_eat = [
        category
        for user_habit in [get_users_ontology_dict()[habit.lower()] for habit in user_profile["habits"]]
        for category in user_habit.doesNotEat] if user_profile else []

    select_node = Node("Food")
    nodes = {"Food": select_node}

    stack = [(select_node, subclass)
             for subclass in list(onto.Food.subclasses())]

    while stack:
        for (parent, item) in stack:
            if item not in user_habits_does_not_eat:
                stack.extend([(item, subclass)
                             for subclass in list(item.subclasses())])
                item_name = item.name.replace("_", " ")
                parent_name = parent.name.replace("_", " ")
                nodes[item_name] = Node(item_name, parent=nodes[parent_name])
            stack.remove((parent, item))

    exporter = DictExporter(dictcls=dict, attriter=sorted)

    structured_class = {}
    for x in test_root.findall(".//{http://www.w3.org/2002/07/owl#}ClassAssertion"):
        main_class = x.find(".//{http://www.w3.org/2002/07/owl#}Class")
        main_class_name = main_class.attrib['IRI'].replace(
            "#", "").replace("_", " ")

        if main_class_name not in nodes:
            continue

        individual = x.find(
            ".//{http://www.w3.org/2002/07/owl#}NamedIndividual")
        individual_name = individual.attrib['IRI'].replace(
            "#", "").replace("_", " ")

        tmp_list = structured_class.get(main_class_name, [])
        tmp_list.append(individual_name)
        structured_class[main_class_name] = tmp_list

    return jsonify({"status": True, "categories": [exporter.export(select_node)], "individuals": structured_class})


@ app.route("/get-user-categories", methods=["POST"])
def get_user_categories():
    parser = ET.XMLParser(remove_blank_text=True)
    test_tree = ET.parse('./FoodMenu.owl', parser)
    test_root = test_tree.getroot()
    tmp_list = []

    whitelist_classes = ["Vegan", "Vegetarian", "Kosher", "Muslim"]

    for x in test_root.findall(".//{http://www.w3.org/2002/07/owl#}SubClassOf"):
        if x.find(".//{http://www.w3.org/2002/07/owl#}ObjectSomeValuesFrom") is not None:
            continue
        classes = x.findall(".//{http://www.w3.org/2002/07/owl#}Class")
        main_class = classes[1].attrib['IRI'].replace("#", "")
        if main_class != "User":
            continue

        sub_class = classes[0].attrib['IRI'].replace("#", "")

        if sub_class in whitelist_classes:
            tmp_list.append(sub_class)

    structured_class = {"User": tmp_list}

    return jsonify({"status": True, "categories": [{"name": "User"}], "individuals": structured_class})
