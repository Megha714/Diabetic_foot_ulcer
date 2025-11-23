from app import app

# Vercel serverless function handler
def handler(request):
    return app(request.environ, request.start_response)
