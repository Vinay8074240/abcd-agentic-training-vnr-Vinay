import math

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


# Cosine search function
def cosine_search(query_vector, data_vectors):
    similarities = []
    
    for idx, vec in enumerate(data_vectors):
        score = cosine_similarity(query_vector, vec)
        similarities.append((idx, score))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


# Example vectors (could be embeddings)
documents = [
    [1, 0, 1, 0],   # Doc 0
    [0, 1, 0, 1],   # Doc 1
    [1, 1, 1, 0],   # Doc 2
    [0, 0, 1, 1]    # Doc 3
]

query = [1, 0, 1, 1]

# Perform cosine search
results = cosine_search(query, documents)

# Display results
print("Cosine Search Results:")
for doc_id, score in results:
    print(f"Document {doc_id} → Similarity: {score:.4f}")
