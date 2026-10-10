"""Long-term memory routes for the signed-in user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.memory.models import UserMemory
from app.memory.service import MemoryService
from app.models.schemas import MemoryCreate, MemoryItem, MemoryUpdate, UserInDB
from app.services.auth.dependencies import get_current_user
from database.sqlite import get_db

router = APIRouter(prefix="/memories", tags=["Memory"])

_LIST_LIMIT = 100
_DEFAULT_IMPORTANCE = 0.5


def _item(memory: UserMemory) -> MemoryItem:
    return MemoryItem(id=memory.id,memory=memory.memory,memory_type=memory.memory_type,importance=memory.importance)


@router.get("", response_model=list[MemoryItem])
def list_memories(current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db),) -> list[MemoryItem]:
    """Return the signed-in user's active memories."""
    rows = MemoryService(db).get_user_memories(user_id=current_user.id, limit=_LIST_LIMIT)
    return [_item(row) for row in rows]


@router.post("", response_model=MemoryItem, status_code=status.HTTP_201_CREATED)
def add_memory(body: MemoryCreate,current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db),) -> MemoryItem:
    """Store a memory the user typed on their profile."""
    created = MemoryService(db).create_memory(user_id=current_user.id,memory=body.memory,memory_type=body.memory_type,importance=_DEFAULT_IMPORTANCE,)
    return _item(created)


@router.put("/{memory_id}", response_model=MemoryItem)
def update_memory(memory_id: str,body: MemoryUpdate,current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db),) -> MemoryItem:
    """Replace one of the signed-in user's memories."""
    service = MemoryService(db)
    current = next((row for row in service.get_user_memories(user_id=current_user.id, limit=_LIST_LIMIT) if row.id == memory_id),None)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    updated = service.update_memory(memory_id=memory_id,memory=body.memory,memory_type=body.memory_type,importance=current.importance,user_id=current_user.id)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    return _item(updated)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(memory_id: str,current_user: UserInDB = Depends(get_current_user),db: Session = Depends(get_db),) -> None:
    """Hide one of the signed-in user's memories."""
    removed = MemoryService(db).delete_memory(user_id=current_user.id, memory_id=memory_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
