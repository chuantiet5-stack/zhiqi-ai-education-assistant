"""消息总线模块"""

from bus.queue import MessageBus, InboundMessage, OutboundMessage

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]