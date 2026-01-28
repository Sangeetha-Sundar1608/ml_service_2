"""Test database models"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_service_2.models.db_models import Base, ProviderUsage, AsyncJob, CostTracking
from ml_service_2.config import settings

# Create engine
engine = create_engine(settings.DATABASE_URL, echo=True)

# Create all tables
print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("✅ Tables created successfully!")

# Test insert
Session = sessionmaker(bind=engine)
session = Session()

# Insert test record
test_usage = ProviderUsage(
    provider="vllm",
    model="llama-3.2-3b",
    prompt_tokens=10,
    completion_tokens=50,
    total_tokens=60,
    latency_ms=1500.5,
    success=True,
    cost_usd=0.0
)

session.add(test_usage)
session.commit()
print("✅ Test record inserted!")

# Query back
records = session.query(ProviderUsage).all()
print(f"✅ Found {len(records)} records")

session.close()