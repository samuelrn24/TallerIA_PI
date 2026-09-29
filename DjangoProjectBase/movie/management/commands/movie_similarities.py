import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

class Command(BaseCommand):
    help = "Compare two movies and optionally a prompt using Hugging Face embeddings"

    def handle(self, *args, **kwargs):
        # ✅ Cargar token de Hugging Face desde el archivo .env
        load_dotenv('../openAI.env')
        client = InferenceClient(token=os.environ.get('HF_TOKEN'))
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # ✅ Cambia estos títulos por películas que tengas en tu base de datos
        try:
            movie1 = Movie.objects.get(title="La captura")
            movie2 = Movie.objects.get(title="Carmencita")
        except Movie.DoesNotExist:
            movies = Movie.objects.all()[:2]
            movie1, movie2 = movies[0], movies[1]

        def get_embedding(text):
            response = client.feature_extraction(text, model=model_name)
            embedding = np.array(response, dtype=np.float32)
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            return embedding

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # ✅ Generar embeddings de ambas películas
        emb1 = get_embedding(movie1.description)
        emb2 = get_embedding(movie2.description)

        # ✅ Calcular similitud entre películas
        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(f"\U0001F3AC Similaridad entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}")

        # ✅ Comparar contra un prompt
        prompt = "película infantil o animada para niños"
        prompt_emb = get_embedding(prompt)

        sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"\U0001F4DD Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
        self.stdout.write(f"\U0001F4DD Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")