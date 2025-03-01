from UltraDict import UltraDict

from insight.zero.component.insight_comp import InsightComponent
from reid_core.reid_comp import ReidComponent

clean_list = ['global', 'SimpleHttp', 'HttpForPhone', 'analysis',
              InsightComponent.SHARED_MEMORY_NAME, ReidComponent.SHARED_MEMORY_NAME]

for i in range(len(clean_list)):
    try:
        UltraDict.unlink_by_name(clean_list[i])
    except Exception as e:
        pass

