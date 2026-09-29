import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

class Command(BaseCommand):
    help = "Generate and store embeddings for all movies using Hugging Face"

    def handle(self, *args, **kwargs):
        load_dotenv('../openAI.env')
        client = InferenceClient(token=os.environ.get('HF_TOKEN'))
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in the database")

        def get_embedding(text):
            response = client.feature_extraction(text, model=model_name)
            embedding = np.array(response, dtype=np.float32)
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            return embedding

        for movie in movies:
            try:
                emb = get_embedding(movie.description)
                movie.emb = emb.tobytes()
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"✅ Embedding stored for: {movie.title}"))
            except Exception as e:
                self.stderr.write(f"❌ Failed to generate embedding for {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS("🎯 Finished generating embeddings for all movies"))