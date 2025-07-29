from database.database import Base, engine
from models.db_models import User, Ticket, Feedback, Escalation

# Create all tables
print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
