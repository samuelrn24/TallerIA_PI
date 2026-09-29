import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Verify embeddings of a random movie"

    def handle(self, *args, **kwargs):
        # Obtener todas las películas que tengan un embedding válido
        movies = [m for m in Movie.objects.all() if m.emb]
        
        if not movies:
            self.stdout.write(self.style.ERROR("No embeddings found in the database."))
            return

        # Seleccionar una película al azar
        random_movie = random.choice(movies)
        
        # Recuperar y transformar el binario de vuelta a un array de numpy
        embedding_vector = np.frombuffer(random_movie.emb, dtype=np.float32)
        
        self.stdout.write(self.style.SUCCESS(f"Película seleccionada al azar: {random_movie.title}"))
        self.stdout.write(f"Longitud del embedding: {len(embedding_vector)} dimensiones")
        self.stdout.write("Primeros 5 valores del vector:")
        self.stdout.write(str(embedding_vector[:5]))