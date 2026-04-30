#!/usr/bin/env python
# coding: utf-8

# # AI-Generated Text Dataset Creation - Few Shot!

# ## Overview
# 
# ### Final Human Dataset Composition (to replicate)
# - **News Articles** – Formal financial & world news (wire-style)
# - **Aeon Essays** – Long-form thoughtful essays (narrative / philosophical)
# - **Yelp Reviews** – Casual, personal, opinionated restaurant reviews
#     - Review distribution: 5 ⭐️: 33%, 4 ⭐️: 35%, 3 ⭐️: 15%, 2 ⭐️: 9%, 1 ⭐️: 7%
# 
# **Target:** ~600 high-quality human samples (balanced across the three sources)
# 
# Formatted: `text, label, generation_type, source`
# 
# ### Generation Approach
# - Model: Local LLM via Ollama
# - Method: Zero-shot prompting (with plans for few-shot comparison)
# - Prompt Design:
#     - Structured templates per writing style (news, essay, review)
#     - Dynamic topic generation using randomized entities and events
#     - Variable constraints, tones, and perspectives to increase diversity
#     - **Includes 2 examplex from the human sample (few-shot)**
# 
# ### Output Labels
# Generation saved with: `source_id, source, text, label, generation_type, topic`
# - label: `ai`
# - generation type: `zero_shot`
# - source: `ai_news`, `ai_essay`, or `ai_review`

# In[24]:


import random
import ollama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time
import os
import csv


# ## Dynamic Variable Setup

# ### News Variables

# In[25]:


# news topic variables

news_entities_business = [
    # specific
    "Amazon", 
    "Apple", 
    "a major European bank", 
    "a global oil company",
    "NVDIA", 
    "Microsoft",
    "the US Federal Reserve",
    "Tesla", 
    "Google", 
    "Meta", 
    "Saudi Aramco", 
    "ExxonMobil",
    "Goldman Sachs", 
    "JPMorgan Chase", 
    "a major pharmaceutical company",
    "a leading semiconductor manufacturer", 
    "a Chinese EV maker",
    "the Federal Reserve", 
    "a major logistics company", 
    "Boeing",
    "a large hedge fund", 
    "a European luxury goods conglomerate"
    "the online retail market",
    "the global energy sector",
    "a multinational corporation",
    "the financial sector",
    "a small tech startup",
    "a major airline"
]

news_entities_sports = [
    # specific
    "the Los Angeles Lakers",
    "the New York Yankees",
    "the Florida Panthers",
    'the US Olympic gymnastics team',
    "Michael Phelps",
    "Serena Williams",
    "Manchester United", 
    "Real Madrid", 
    "the Golden State Warriors",
    "the Dallas Cowboys", 
    "LeBron James", 
    "Lionel Messi", 
    "Novak Djokovic",
    "a top-ranked tennis player", 
    "the Boston Celtics", 
    "a rising UFC fighter",
    "the Chicago Blackhawks",
    "an English Premier League club",
    "the Brazilian national football team"
    "a professional ice hockey team",
    "the NFL",
    "an Olympic swimmer",
    "a championship contender",
    "a basketball draft pick"
    "an Olympic snowboarder",
    "a World Cup soccer team",
]

news_events_business = [
    "reporting quarterly earnings",
    "announcing a merger",
    "facing declining revenue",
    "expanding into international markets",
    "responding to new regulations",
    "launching a new product",
    "adjusting to supply chain disruptions",
    "cutting thousands of jobs", 
    "announcing major layoffs",
    "securing a huge government contract", 
    "facing a major lawsuit",
    "going public via IPO", 
    "being acquired by a competitor",
    "launching an aggressive cost-cutting plan", 
    "entering the AI race",
    "dealing with activist investors", 
    "expanding into renewable energy",
]

news_events_sports = [
    "winning a championship",
    "losing a critical game/meet",
    "being traded to another team",
    "facing a controversial decision",
    "recovering from an injury",
    "struggling during the season",
    "upsetting a favored opponent",
    "a coach being fired or hired",
    "signing a record-breaking contract",
    "being suspended for controversial behavior",
    "making a surprise comeback",
    "breaking a long-standing record",
    "missing the playoffs",
    "being eliminated in the quarterfinals",
    "undergoing major surgery and recovery",
    "clashing with team management",
    "winning a major endorsement deal",
    "retiring after a legendary career"
]


# In[26]:


def build_news_topic():
    r = random.random()

    if r < 0.60: # 60% business
        entity = random.choice(news_entities_business)
        event = random.choice(news_events_business)
        topic = f"{entity} {event}"

    elif r < 0.90: # 30% sports
        entity = random.choice(news_entities_sports)
        event = random.choice(news_events_sports)
        topic = f"{entity} {event}"

    else: # 10% mixed or more complex topics
        entity1 = random.choice(news_entities_business + news_entities_sports)
        entity2 = random.choice(news_entities_business + news_entities_sports)
        event = random.choice(news_events_business + news_events_sports)
        topic = f"{entity1} and {entity2} {event}"

    return topic


# In[27]:


# other news generation variables
news_styles = [
    "Write a news report",
    "Report on recent developments",
    "Write a short news dispatch",
    "Summarize a recent event",
    "Cover a developing story",
    "Provide a brief news update",
    "Write a breaking news-style article",
    "Write a straightforward wire-style report",
    "Write an analytical news piece",
    "Produce a neutral, fact-focused dispatch"
]

news_angles = [
    "focus on economic impact",
    "focus on regulatory hurdles",
    "focus on business strategy",
    "focus on market reactions",
    "focus on public response",
    "focus on policy implications",
    "focus on international context",
    "focus on investor sentiment",
    "focus on long-term consequences",
    "focus on fan or consumer reaction",
    "focus on competitive landscape",
    "focus on technological disruption",
    "focus on ethical concerns",
    "focus on short-term vs long-term effects"
]

news_locations = [
    "NEW YORK", "LONDON", "HONG KONG", "TOKYO",
    "SINGAPORE", "DUBAI", "SYDNEY", "BERLIN", 
    "PARIS", "BEIJING", "MUMBAI", "MOSCOW", "TORONTO",
    "JOHANNESBURG", "RIYADH", "SEOUL", "MEXICO CITY",
    "LOS ANGELES", "CHICAGO", "SAN FRANCISCO", "WASHINGTON D.C.",
    "SEATTLE", "ATLANTA", "MIAMI", "DALLAS", "BOSTON"
]

news_constraints = [
    "Use slightly shorter sentences.",
    "Include at least one numerical statistic.",
    "Keep the tone slightly neutral and detached.",
    "Avoid overly complex explanations.",
    "Let the writing feel slightly compressed.",
    "Do not over-explain context.",
    "Allow some abrupt transitions.",
    "Keep the pacing quick and information-dense.",
    "Include one short quote from a key person.",
    "Mention one specific number or statistic.",
    "Use slightly informal phrasing in one or two places.",
    "End with a forward-looking sentence.",
    "Keep some sentences under 15 words.",
    "Include a minor detail about location if it fits.",
    "Vary paragraph lengths.",
    "Avoid perfect symmetry in structure."
]


# ### Essays Variables

# In[28]:


# primary essay topic variables

essay_entities = {

    "stories and literature": [
        "the role of storytelling in shaping human experience",
        "how literature reflects cultural identity",
        "the meaning of narrative in understanding reality",
        "the power of unreliable narrators in fiction",
        "how myths and legends shape collective identity",
        "the evolution of the hero's journey in modern literature",
        "literature as a form of resistance and protest",
        "the blurred line between fiction and autobiography",
        "how George Orwell's 1984 continues to shape modern political thought",
        "the influence of Toni Morrison's novels on American identity",
        "how Franz Kafka's stories explore alienation and bureaucracy",
        "the role of Chimamanda Ngozi Adichie in contemporary African literature",
        "how Margaret Atwood's dystopian works reflect current social anxieties"
    ],

    "history": [
        "how historical events influence modern society",
        "the relationship between memory and historical truth",
        "the way societies reinterpret the past over time",
        "how collective trauma influences national identity",
        "the dangers of romanticizing historical figures",
        "how technology changes the way we record and remember history",
        "the role of forgotten voices in historical narratives",
        "how Winston Churchill's leadership during WWII is remembered and debated",
        "the legacy of Nelson Mandela and the politics of reconciliation",
        "how Howard Zinn's 'A People's History' changed American historical narratives",
        "the historical impact of the Civil Rights Movement leaders like Martin Luther King Jr."
    ],

    "thinkers and theories": [
        "the influence of philosophical theories on human behavior",
        "how major thinkers shape modern intellectual life",
        "the role of abstract theory in everyday decision-making",
        "how Nietzsche's ideas continue to influence modern culture",
        "the lasting impact of existentialist thought",
        "how Eastern and Western philosophy intersect today",
        "the relevance of ancient Greek thinkers in the digital age",
        "the continuing influence of Michel Foucault on power and knowledge",
        "how Hannah Arendt's ideas on totalitarianism remain relevant today",
        "the philosophical legacy of Simone de Beauvoir and existential feminism",
        "how John Rawls' theory of justice shapes modern political philosophy"
    ],

    "ethics": [
        "ethical responsibility in modern society",
        "the tension between individual freedom and moral obligation",
        "how people navigate difficult moral dilemmas",
        "ethical questions raised by artificial intelligence",
        "the morality of genetic engineering and designer babies",
        "personal responsibility in the age of social media",
        "the ethics of climate change and future generations",
        "the role of empathy in ethical decision-making",
    ],

    "biology": [
        "the relationship between biology and human behavior",
        "how evolutionary processes shape decision-making",
        "the biological basis of human instincts and emotions",
        "how genetics influence personality traits",
        "the biological roots of empathy and altruism",
        "the impact of neuroplasticity on human potential",
        "how sleep and dreams affect decision-making"
    ],

    "history of ideas": [
        "how Karl Marx's ideas continue to influence 21st-century politics",
        "how ideas evolve across generations",
        "the transmission of knowledge through time",
        "how intellectual traditions influence modern thinking",
        "how the concept of individualism has changed over centuries",
        "the rise and fall of utopian thinking",
        "how scientific revolutions reshape philosophical worldviews"
    ],

    "human rights and justice": [
        "the challenge of enforcing human rights globally",
        "justice in unequal societies",
        "the balance between law and moral fairness",
        "the tension between security and civil liberties",
        "justice and forgiveness in post-conflict societies",
        "the effectiveness of international courts and tribunals",
        "economic inequality as a human rights issue",
        "the global impact of Malala Yousafzai's activism for girls' education",
        "how Ruth Bader Ginsburg advanced gender equality through the courts",
        "the legacy of Desmond Tutu and South Africa's Truth and Reconciliation Commission",
        "the work of human rights activist Ai Weiwei in challenging authoritarianism",
    ],

    "cognition and intelligence": [
        "the limits of human intelligence",
        "how perception shapes understanding",
        "the relationship between knowledge and belief",
        "the difference between intelligence and wisdom",
        "how bias affects human reasoning",
        "the role of intuition versus logical thinking",
        "can machines ever truly understand",
        "how Alan Turing's ideas about machine intelligence still shape AI debates",
    ],

    "conscience and altered state": [
        "the nature of conscious experience",
        "how altered states affect perception and identity",
        "the role of awareness in shaping reality",
        "the philosophical implications of lucid dreaming",
        "how meditation changes our sense of self",
        "the nature of near-death experiences",
        "psychedelics and the boundaries of consciousness"
    ],

    "art": [
        "the role of art in society",
        "why humans create and value art",
        "how artistic expression reflects human emotion",
        "the relationship between art and political power",
        "why some art becomes timeless while most is forgotten",
        "the commercialization of artistic expression",
        "art as therapy and its psychological benefits",
        "how Pablo Picasso's Guernica became a symbol of anti-war protest",
        "the cultural significance of Frida Kahlo's self-portraits and identity",
        "how Banksy's street art challenges power structures and capitalism",
        "the provocative work of Marina Abramović and performance art",
    ]
}

essay_events = [
    "shapes modern society",
    "influences human behavior",
    "creates ethical dilemmas",
    "limits human understanding",
    "changes over time",
    "is often misunderstood",
    "reveals deeper truths about human life",
    "challenges traditional assumptions",
    "raises uncomfortable questions",
    "highlights contradictions in human nature",
    "bridges the gap between theory and lived experience",
    "forces us to reconsider our values",
    "reveals the limits of language",
    "explores the tension between freedom and belonging"
]


# In[29]:


def build_essay_topic():
    # pick category
    category = random.choice(list(essay_entities.keys()))

    # pick main entity/topic from that category
    entity = random.choice(essay_entities[category])

    # add variation with events/angles
    r = random.random()

    if r < 0.55: # ~55% simple + clean topic
        topic = entity

    elif r < 0.85: # ~30% classic format
        event = random.choice(essay_events)
        topic = f"{entity} and how it {event}"

    else: # ~15% more complex / interesting combinations
        event1 = random.choice(essay_events)
        event2 = random.choice(essay_events)
        while event2 == event1: # avoid duplicates
            event2 = random.choice(essay_events)

        topic = f"{entity}, {event1}, and {event2}"

    return topic


# In[30]:


# other essay generation variables
essay_styles = [
    "Write a reflective passage",
    "Discuss the idea of",
    "Explore the concept of",
    "Analyze the theme of",
    "Consider the implications of",
    "Examine the role of",
    "Think through the idea of",
    "Reflect on the broader implications of",
    "Delve into the complexities surrounding",
    "Offer a personal meditation on",
    "Question long-held beliefs about",
    "Investigate the subtle ways",
]

essay_tones = [
    "slightly reflective",
    "analytical",
    "thoughtful but informal",
    "philosophical",
    "critical but calm",
    "curious and exploratory",
    "measured and observational",
    "introspective and personal",
    "skeptical yet open-minded",
    "conversational and thoughtful",
    "quietly provocative",
    "nuanced and balanced",
]

essay_constraints = [
    "Allow some sentences to be longer and more complex.",
    "Avoid overly structured arguments.",
    "Let the writing feel like part of an ongoing discussion.",
    "Do not fully resolve the argument.",
    "Allow small tangents or side thoughts.",
    "Keep the tone slightly uneven rather than perfectly polished.",
    "Do not try to summarize at the end.",
    "Include one short personal reflection or anecdote.",
    "Use occasional rhetorical questions.",
    "Vary sentence rhythm - mix short and long sentences.",
    "Leave some ideas slightly unresolved.",
    "Avoid grand conclusions or final answers.",
    "Sound like a thoughtful person thinking out loud.",
    "Include one moment of doubt or uncertainty.",
]

essay_openings = [
    "Begin mid-thought rather than with a formal introduction.",
    "Start with a general observation.",
    "Start with a question.",
    "Start with a reflective statement.",
    "Start with a reference to a broad idea.",
    "Start with a small personal observation.",
    "Begin with a surprising contradiction.",
    "Open with a memory or everyday example.",
    "Start with a direct address to the reader.",
    "Begin in the middle of a thought.",
]


# ### Review Variables

# In[31]:


# primary review variables
review_entities = [
    # specific-ish
    "a busy Italian restaurant",
    "a small vegan cafe",
    "a popular sushi spot",
    "a high-end steakhouse",
    "McDonald's",
    "Culver's",
    "a local pizza place",
    "The Cheesecake Factory",
    "Olive Garden",
    "a hole-in-the-wall taco truck", 
    "a fancy French bistro", 
    "a Korean BBQ restaurant",
    "a popular brunch cafe", 
    "a seafood restaurant by the beach", 
    "a Thai street food spot",
    "a Mexican taqueria", 
    "a donut shop", 
    "a craft brewery with food", 
    "a ramen bar",
    "a Southern soul food restaurant", 
    "a dim sum palace", 
    "a food truck park",
    "a local restaurant",
    "a family owned diner",
    "a neighborhood cafe",
    "a chain restaurant",
    "a newly opened restaurant",
    "a breakfast spot",
    "a lunch spot",
    "a dinner spot",
    "fine dining experience",
    "large buffet-style restaurant",
    "trendy and popular new eatery",
    "a trendy juice bar", 
    "a neighborhood bookstore cafe",
    "a hair salon", 
    "a boutique gym or yoga studio", 
    "a car wash", 
    "a pet grooming salon",
    "an ice cream parlor", 
    "a hardware store", 
    "a farmers market stall", 
    "a nail salon",
    "a movie theater", 
    "a massage spa", 
    "a local bakery"
]
review_events = [
    "during a busy weekend",
    "during the slow season",
    "on a quiet weekday",
    "on a busy weekday",
    "during a special event",
    "on a rainy day",
    "on a sunny day",
    "during the lunch rush",
    "during the dinner rush",
    "on a specific holiday",
    "during a local festival",
    "as take-out service",
    "during a first date", 
    "on a family outing",
    "after a long workday",
    "for a birthday celebration", 
    "during a group hangout", 
    "on a solo visit",
    "while on vacation", 
    "right after it just opened", 
    "during happy hour",
    "on a freezing cold day", 
    "during a heatwave", 
    "for takeout on a weeknight",
    "for delivery during bad weather", 
    "as a quick stop before a movie",
]


# In[32]:


def build_review_topic():
    entity = random.choice(review_entities)
    event = random.choice(review_events)
    return f"{entity} visited {event}"


# In[33]:


# other review generation variables

review_styles = [
    "Write a review",
    "Describe your experience at",
    "Share your thoughts on",
    "Write about a recent visit to",
    "Explain your experience at",
    "Give your opinion of",
    "Write a casual Yelp-style review about",
    "Share an honest experience at",
    "Give your unfiltered thoughts on visiting",
    "Describe what happened when you went to",
    "Write a mixed review of",
]

review_perspectives = [
    "as a first-time customer",
    "as a regular customer",
    "as someone visiting for a special occasion",
    "as someone trying it casually",
    "as someone with high expectations",
    "as someone with generic opinions about the cuisine",
    "as someone who is easily satisfied",
    "as an impatient customer",
    "as someone who was skeptical at first"
    "as a grease-loving eater",
    "as a health-conscious eater",
    "as a picky eater",
    "as a very hungry person", 
    "as someone on a budget", 
    "as a food snob",
    "as a parent with kids", 
    "as someone with dietary restrictions",
    "as a tourist in the area",
    "as a local who rarely eats out",
    "as someone who hates waiting", 
    "as a big tipper who values service",
    "as a vegetarian/vegan customer", 
    "as someone recovering from surgery"
]

review_details = [
    "mention food quality",
    "mention service speed",
    "mention staff behavior",
    "mention atmosphere",
    "mention pricing",
    "mention wait times",
    "mention cleanliness",
    "mention specific menu items",
    "mention portion sizes", 
    "mention noise level", 
    "mention parking situation",
    "mention bathroom cleanliness", 
    "mention background music or vibe",
    "mention value for money", 
    "mention how busy it was", 
    "mention outdoor seating",
    "mention a specific server or employee", 
    "mention temperature of the food",
    "mention packaging quality for takeout/leftovers"
]

review_constraints = [
    "Allow informal phrasing and slight grammar imperfections.",
    "Include at least two specific details.",
    "Let the tone fluctuate slightly.",
    "Avoid making the structure too neat.",
    "Give a tangent or side comment.",
    "Do not make the review overly polished.",
    "Include a minor contradiction if natural.",
    "Mention a specific dish or drink",
    "Discuss a specific staff interaction",
    "Include a specific allergy or dietary preference",
    "Use at least one contraction (I'm, it's, don't, etc.).",
    "Add a short personal story or reason for visiting.",
    "Vary sentence length — include some very short sentences.",
    "End with a clear recommendation or warning.",
    "Mention the weather or time of day if it fits naturally.",
    "Be slightly repetitive or ramble in one sentence.",
]


# ### Additional Helpers

# In[34]:


# star rating distribution

def sample_star():
    return random.choices(
        [1,2,3,4,5],
        weights=[7,9,15,35,33], # based on the human data
        k=1
    )[0]


# In[35]:


# length randomization
min_options = [50, 95, 160, 210, 300]
max_options = [350, 425, 540, 750]

def sample_length(min_options, max_options):
    return random.choice(min_options), random.choice(max_options)


# In[36]:


# list sampling

def sample_list(items, min_k=1, max_k=4):
    k = random.randint(min_k, min(max_k, len(items)))
    return random.sample(items, k)


# ## Few Shot Prompt Example Collection

# In[37]:


human_df = pd.read_csv("../data/human_dataset.csv")

# create separate pools by source
human_pools = {
    "news":   human_df[human_df["source"] == "articles.csv"]["text"].tolist(),
    "essay":  human_df[human_df["source"] == "essays_texts.csv"]["text"].tolist(),
    "review": human_df[human_df["source"] == "yelp.csv"]["text"].tolist()
}

print({k: len(v) for k, v in human_pools.items()})


# In[38]:


# function to get few-shot examples by category

def get_few_shot_examples(category: str, n: int = 2):
    pool = human_pools[category]
    return random.sample(pool, min(n, len(pool)))


# ## Prompts

# In[39]:


def news_prompt(topic):
    style = random.choice(news_styles)
    angle = random.choice(news_angles)
    location = random.choice(news_locations)
    min_len, max_len = sample_length(min_options, max_options)

    constraints = sample_list(news_constraints, 1, 3)
    constraint_text = "\n".join([f"- {c}" for c in constraints])

    few_shot_examples = examples = get_few_shot_examples("news", n=2)
    few_shot_examples_block = "\n\n".join([f"--- REAL HUMAN NEWS ARTICLE ---\n{ex}\n--- END ---\n" for ex in examples])

    return f"""
{style} ({min_len}-{max_len} words) about {topic}.

Here are 3 real examples of your writing style to reference:
{few_shot_examples_block}

Requirements:
- You are a professional journalist writing for a major news outlet.
- Do NOT reference events after 2021
- Start with an all capitalized location header (e.g., "{location}:")
- {angle}
- {constraint_text}
- Include realistic details, numbers, names, and context (made up but plausible)
- Avoid overly polished or academic phrasing
- Keep writing factual and slightly restrained
- Do NOT include a conclusion paragraph
- Mimic human writing so that it is not detectable as AI-generated
"""


# In[40]:


def essay_prompt(topic):
    style = random.choice(essay_styles)
    tone = random.choice(essay_tones)
    opening = random.choice(essay_openings)
    min_len, max_len = sample_length(min_options, max_options)

    constraints = sample_list(essay_constraints, 1, 4)
    constraint_text = "\n".join([f"- {c}" for c in constraints])

    few_shot_examples = examples = get_few_shot_examples("essay", n=2)
    few_shot_examples_block = "\n\n".join([f"--- REAL HUMAN ESSAY ---\n{ex}\n--- END ---\n" for ex in examples])

    return f"""
{style} ({min_len}-{max_len} words) about {topic}.

Here are 3 real examples of your writing style to reference:
{few_shot_examples_block}

Requirements:
- You are a real, ordinary human writer exploring complex ideas.
- The tone should feel {tone}
- {opening}
- {constraint_text}
- No title or header
- Write as if this is part of a longer essay, not a complete piece
- Use varied sentence lengths. Occasionally be casual or blunt. 
- Do NOT include a conclusion - let the writing be unfinished or exploratory
- Avoid overly polished, flowery, poetic, or perfectly structured phrasing
- Mimic human writing so that it is not detectable as AI-generated
"""



# In[41]:


def review_prompt(topic, stars):
    style = random.choice(review_styles)
    perspective = random.choice(review_perspectives)

    details = sample_list(review_details, 1, 3)
    detail_text = "\n".join([f"- {d}" for d in details])

    constraints = sample_list(review_constraints, 1, 3)
    constraint_text = "\n".join([f"- {c}" for c in constraints])

    few_shot_examples = examples = get_few_shot_examples("review", n=2)
    few_shot_examples_block = "\n\n".join([f"--- REAL HUMAN REVIEW ---\n{ex}\n--- END ---\n" for ex in examples])

    tone_map = {
        5: "very positive",
        4: "mostly positive",
        3: "mixed",
        2: "mostly negative",
        1: "very negative"
    }

    min_len, max_len = sample_length(min_options, max_options)

    return f"""
{style} ({min_len}-{max_len} words) about a {topic}, {perspective}.

Here are 3 real examples of your writing style to reference:
{few_shot_examples_block}

Requirements:
- You are a customer sharing your experience on Yelp.
- First-person perspective
- The tone should feel {tone_map[stars]}
- Match a {stars}-star rating
- Include specific details and personal experiences.
- Allow slight tangents or informal phrasing.
- Do NOT use emojis or UTF-8 characters.
- {detail_text}
- {constraint_text}
- Avoid exaggerated or overly playful phrasing
- Mimic human writing so that it is not detectable as AI-generated
"""


# ## LLM Setup

# In[42]:


MODEL = "mistral:instruct"
N_WORKERS = 3 # testing on 3

def generate_one(task):
    source_name = task["source"]
    topic = task["topic"]
    prompt = task["prompt"]
    i = task["i"]

    start = time.time()

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                'temperature': 0.75,
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1
            }
        )

        text = response["message"]["content"]
        error = None

    except Exception as e:
        text = None
        error = str(e)

    end = time.time()

    return {
        "source_id": f"{source_name}_{i}",
        "source": source_name,
        "text": text,
        "label": "ai",
        "generation_type": "zero_shot",
        "topic": topic,
        "latency_sec": round(end - start, 2),
        "error": error
    }


# In[43]:


# build tasks

def build_tasks(n_per_type=5):
    tasks = []

    # separate counters per source (cleaner IDs)
    counters = {
        "ai_news": 0,
        "ai_essay": 0,
        "ai_review": 0
    }

    for _ in range(n_per_type):

        # news
        topic = build_news_topic()

        tasks.append({
            "i": counters["ai_news"],
            "source": "ai_news",
            "topic": topic,
            "prompt": news_prompt(topic)
        })

        counters["ai_news"] += 1


        # essay
        topic = build_essay_topic()

        tasks.append({
            "i": counters["ai_essay"],
            "source": "ai_essay",
            "topic": topic,
            "prompt": essay_prompt(topic)
        })

        counters["ai_essay"] += 1


        # review
        topic = build_review_topic()
        stars = sample_star()

        tasks.append({
            "i": counters["ai_review"],
            "source": "ai_review",
            "topic": topic,
            "stars": stars,
            "prompt": review_prompt(topic, stars)
        })

        counters["ai_review"] += 1

    # shuffle tasks to avoid grouped generation
    random.shuffle(tasks)

    return tasks


# In[44]:


# test generations on a batch
# build tasks dynamically
tasks = build_tasks(n_per_type=5)

start_total = time.time()
results = []

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(generate_one, task): task for task in tasks}

    for i, future in enumerate(as_completed(futures)):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            # catch any unexpected errors
            results.append({
                "source": futures[future]["source"],
                "source_id": f"error_{i}",
                "text": None,
                "label": "ai",
                "generation_type": "zero_shot",
                "topic": futures[future]["topic"],
                "latency_sec": None,
                "error": str(e)
            })

        # progress tracking
        if (i + 1) % 5 == 0:
            print(f"{i+1}/{len(tasks)} completed")

end_total = time.time()

# convert to dataframe
ai_test_df = pd.DataFrame(results)

# summary
print(f"\nTotal time: {round(end_total - start_total, 2)} seconds")
print("\nLatency stats:")
print(ai_test_df["latency_sec"].describe())

print("\nErrors:")
print(ai_test_df["error"].value_counts(dropna=False))

ai_test_df.head()


# In[45]:


def print_examples_by_source(df, n=2):
    for source in df["source"].unique():
        print(f"\n===== {source.upper()} EXAMPLES =====")

        subset = df[df["source"] == source].head(n)

        for i, row in subset.iterrows():
            print(f"\n--- Topic: {row['topic']} ---")
            print(row["text"])
            print("-----")

print_examples_by_source(ai_test_df, n=2)


# ## Batch Generate 600 Samples

# In[ ]:


# set output file

OUTPUT_FILE = "../data/ai_generated_fewshot.csv"

# load existing data if present (resume capability)
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    print(f"Resuming: {len(existing_df)} rows already exist")
else:
    # create file with header if it doesn't exist
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_id", "source", "text",
            "label", "generation_type",
            "topic", "latency_sec", "error"
        ])


# In[22]:


# function to append/save rows during generation 

def append_row_safe(row):
    try:
        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                row.get("source_id"),
                row.get("source"),
                row.get("text"),
                row.get("label"),
                row.get("generation_type"),
                row.get("topic"),
                row.get("latency_sec"),
                row.get("error")
            ])
    except Exception as e:
        print(f"Failed to write row: {e}")


# In[23]:


# generation wrapper

def generate_one_safe(task):
    import time

    start = time.time()

    try:
        result = generate_one(task)
        result["error"] = None

    except Exception as e:
        result = {
            "source_id": f"{task['source']}_{task['i']}",
            "source": task["source"],
            "text": None,
            "label": "ai",
            "generation_type": "zero_shot",
            "topic": task["topic"],
            "latency_sec": None,
            "error": str(e)
        }

    end = time.time()
    result["latency_sec"] = round(end - start, 2)

    # save immediately!!!!!!!
    append_row_safe(result)

    return result


# In[24]:


def run_batch(n_per_type=25, workers=2):
    tasks = build_tasks(n_per_type=n_per_type)

    print(f"\nStarting batch ({len(tasks)} tasks)...")

    start_total = time.time()
    results = []
    error_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_one_safe, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            task = futures[future]

            try:
                # timeout protection
                result = future.result(timeout=120)

            except Exception as e:
                result = {
                    "source_id": f"{task['source']}_{task['i']}",
                    "source": task["source"],
                    "text": None,
                    "label": "ai",
                    "generation_type": "zero_shot",
                    "topic": task["topic"],
                    "latency_sec": None,
                    "error": str(e)
                }

                print(f"Error: {task['source']} | {task['topic']}")

            if result.get("error") is not None:
                error_count += 1

            results.append(result)

            if (i + 1) % 5 == 0:
                print(f"{i+1}/{len(tasks)} completed")

    end_total = time.time()

    print(f"\nBatch done in {round(end_total - start_total, 2)} seconds")
    print(f"Errors in batch: {error_count}")

    return results, error_count


# In[25]:


# 600 generation loop

all_results = []

NUM_BATCHES = 8   # ~600 samples (8 × 75)
MAX_ERRORS = 20

for b in range(NUM_BATCHES):
    print(f"\n========================")
    print(f"Running batch {b+1}/{NUM_BATCHES}")
    print(f"========================")

    batch_results, batch_errors = run_batch(n_per_type=25, workers=N_WORKERS)

    all_results.extend(batch_results)

    # STOP if too many errors
    if batch_errors > MAX_ERRORS:
        print("Too many errors, stopping early!!")
        break

    # cooling break
    time.sleep(45)


# In[ ]:




