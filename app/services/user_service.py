import os
import sqlite3
import datetime
from typing import Optional, List, Dict, Any
from loguru import logger
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError

from app.models.user import UserModel, UserRole, UserStatus
from app.core.security import hash_password


class UserService:
    """
    Manages users (HOD and Class Advisors) with dual MongoDB & resilient SQLite storage:
    - Primary: MongoDB `users` collection if connected
    - Resilient Fallback: `smartclass.sqlite` `users` table
    - Automatically synchronizes across both layers
    - Guarantees HOD bootstrap account creation on initial startup
    """

    def __init__(self, db_path: Optional[str] = None, mongo_uri: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = db_path or os.path.join(base_dir, "database", "smartclass.sqlite")
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("DATABASE_NAME", "smartclass_vision_ai")

        self.mongo_client: Optional[MongoClient] = None
        self.mongo_users_col = None
        self.mongo_online = False

        # 1. Initialize local SQLite table
        self._init_sqlite()

        # 2. Try connecting to MongoDB (non-blocking, won't crash if offline)
        self._init_mongodb()

        # 3. Bootstrap default HOD account
        self._bootstrap_hod_account()

    def _init_sqlite(self):
        """Initializes SQLite users table for persistent storage."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    department TEXT,
                    year TEXT,
                    section TEXT,
                    assigned_classroom TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.commit()

    def _init_mongodb(self):
        """Attempts connection to MongoDB and creates unique index on user_id."""
        try:
            self.mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=1500)
            # Ping database to verify connection
            self.mongo_client.admin.command('ping')
            db = self.mongo_client[self.db_name]
            self.mongo_users_col = db["users"]
            self.mongo_users_col.create_index([("user_id", ASCENDING)], unique=True)
            self.mongo_online = True
            logger.info(f"MongoDB successfully connected at {self.mongo_uri}, database: {self.db_name}")
        except Exception as e:
            self.mongo_online = False
            self.mongo_client = None
            self.mongo_users_col = None
            logger.warning(f"MongoDB offline or unreachable ({e}). Operating in resilient local storage mode.")

    def _bootstrap_hod_account(self):
        """Ensures at least one default HOD account exists for administrative access."""
        default_hod_id = os.getenv("DEFAULT_HOD_USER_ID", "HOD001")
        default_hod_pass = os.getenv("DEFAULT_HOD_PASSWORD", "admin123")
        default_hod_name = os.getenv("DEFAULT_HOD_NAME", "Head of Department")

        existing = self.get_user_by_id(default_hod_id)
        if not existing:
            hashed = hash_password(default_hod_pass)
            now = datetime.datetime.utcnow().isoformat()
            hod = UserModel(
                user_id=default_hod_id,
                name=default_hod_name,
                role=UserRole.HOD,
                password_hash=hashed,
                email="hod.aids@college.edu",
                department="AI&DS",
                status=UserStatus.ACTIVE,
                created_at=now,
                updated_at=now
            )
            self.create_user(hod)
            logger.info(f"Bootstrapped default HOD account '{default_hod_id}'.")

    def get_user_by_id(self, user_id: str) -> Optional[UserModel]:
        """Retrieves a user by user_id from MongoDB (if online) or SQLite."""
        uid = str(user_id).strip()
        # 1. Try MongoDB
        if self.mongo_online and self.mongo_users_col is not None:
            try:
                doc = self.mongo_users_col.find_one({"user_id": uid})
                if doc:
                    doc.pop("_id", None)
                    return UserModel(**doc)
            except Exception as e:
                logger.debug(f"MongoDB read error: {e}")

        # 2. SQLite lookup
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE user_id = ? LIMIT 1;", (uid,))
                row = cur.fetchone()
                if row:
                    return UserModel(**dict(row))
        except Exception as e:
            logger.error(f"SQLite user query error: {e}")

        return None

    def get_user_by_id_or_email(self, identifier: str) -> Optional[UserModel]:
        """Retrieves a user by user_id OR email from MongoDB or SQLite (case-insensitive)."""
        ident = str(identifier).strip()
        # 1. Try MongoDB
        if self.mongo_online and self.mongo_users_col is not None:
            try:
                doc = self.mongo_users_col.find_one({"$or": [{"user_id": ident}, {"email": {"$regex": f"^{ident}$", "$options": "i"}}]})
                if doc:
                    doc.pop("_id", None)
                    return UserModel(**doc)
            except Exception as e:
                logger.debug(f"MongoDB read error: {e}")

        # 2. SQLite lookup
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE user_id = ? OR email = ? COLLATE NOCASE LIMIT 1;", (ident, ident))
                row = cur.fetchone()
                if row:
                    return UserModel(**dict(row))
        except Exception as e:
            logger.error(f"SQLite user query error: {e}")

        return None

    def create_user(self, user: UserModel) -> UserModel:
        """Persists a new user. Rejects duplicates."""
        existing = self.get_user_by_id(user.user_id)
        if existing:
            raise ValueError(f"User ID '{user.user_id}' already exists.")

        user_dict = user.model_dump()

        # 1. Persist in SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO users (
                    user_id, name, role, password_hash, email, phone,
                    department, year, section, assigned_classroom, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                user.user_id, user.name, user.role.value, user.password_hash,
                user.email, user.phone, user.department, user.year, user.section,
                user.assigned_classroom, user.status.value, user.created_at, user.updated_at
            ))
            conn.commit()

        # 2. Persist in MongoDB if online
        if self.mongo_online and self.mongo_users_col is not None:
            try:
                self.mongo_users_col.insert_one(user_dict.copy())
            except Exception as e:
                logger.warning(f"MongoDB insert error for user '{user.user_id}': {e}")

        return user

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[UserModel]:
        """Updates user profile attributes."""
        user = self.get_user_by_id(user_id)
        if not user:
            return None

        # Disallow updating primary key and password directly here
        updates.pop("user_id", None)
        updates.pop("password_hash", None)
        updates["updated_at"] = datetime.datetime.utcnow().isoformat()

        # Update in SQLite
        set_clauses = []
        values = []
        for k, v in updates.items():
            if v is not None:
                set_clauses.append(f"{k} = ?")
                values.append(v)

        if set_clauses:
            values.append(user_id)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = ?;", values)
                conn.commit()

        # Update in MongoDB
        if self.mongo_online and self.mongo_users_col is not None:
            try:
                self.mongo_users_col.update_one({"user_id": user_id}, {"$set": updates})
            except Exception as e:
                logger.warning(f"MongoDB update error: {e}")

        return self.get_user_by_id(user_id)

    def update_status(self, user_id: str, status: str) -> bool:
        """Enables or disables a user account."""
        clean_status = status.lower().strip()
        if clean_status not in [UserStatus.ACTIVE.value, UserStatus.DISABLED.value]:
            raise ValueError(f"Invalid status '{status}'. Must be 'active' or 'disabled'.")

        now = datetime.datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE users SET status = ?, updated_at = ? WHERE user_id = ?;", (clean_status, now, user_id))
            conn.commit()

        if self.mongo_online and self.mongo_users_col is not None:
            try:
                self.mongo_users_col.update_one({"user_id": user_id}, {"$set": {"status": clean_status, "updated_at": now}})
            except Exception as e:
                logger.warning(f"MongoDB status update error: {e}")

        return True

    def update_password(self, user_id: str, new_password_hash: str) -> bool:
        """Updates user's hashed password."""
        now = datetime.datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE user_id = ?;", (new_password_hash, now, user_id))
            conn.commit()

        if self.mongo_online and self.mongo_users_col is not None:
            try:
                self.mongo_users_col.update_one({"user_id": user_id}, {"$set": {"password_hash": new_password_hash, "updated_at": now}})
            except Exception as e:
                logger.warning(f"MongoDB password update error: {e}")

        return True

    def list_advisors(self) -> List[UserModel]:
        """Lists all Class Advisors."""
        advisors: List[UserModel] = []
        if self.mongo_online and self.mongo_users_col is not None:
            try:
                docs = self.mongo_users_col.find({"role": UserRole.CLASS_ADVISOR.value}).sort("name", 1)
                for d in docs:
                    d.pop("_id", None)
                    advisors.append(UserModel(**d))
                return advisors
            except Exception as e:
                logger.debug(f"MongoDB list advisors error: {e}")

        # SQLite fallback
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE role = 'class_advisor' ORDER BY name ASC;")
            for row in cur.fetchall():
                advisors.append(UserModel(**dict(row)))

        return advisors

    def get_all_users(self) -> List[UserModel]:
        """Lists all users across roles."""
        users: List[UserModel] = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM users ORDER BY role ASC, name ASC;")
            for row in cur.fetchall():
                users.append(UserModel(**dict(row)))
        return users
