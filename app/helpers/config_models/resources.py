"""
静态资源配置
"""

from pydantic import BaseModel, Field


class ResourcesModel(BaseModel):
    """
    静态资源配置
    """
    public_url: str = Field(default="http://localhost:8080/public")
