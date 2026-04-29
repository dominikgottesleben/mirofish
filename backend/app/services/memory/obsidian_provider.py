"""
Obsidian-Markdown-Memory-Provider
Speichert den Wissensgraphen lokal als Markdown-Dateien in einer Obsidian-Vault-Struktur
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .base import MemoryProvider, MemoryNode, MemoryEdge, MemorySearchResult
from ...config import Config
from ...utils.logger import get_logger
from ...utils.llm_client import LLMClient

logger = get_logger('mirofish.memory.obsidian')

class ObsidianMemoryProvider(MemoryProvider):
    """Memory-Provider für lokale Markdown-Dateien (Obsidian-kompatibel)"""
    
    def __init__(self, vault_root: Optional[str] = None):
        # Basis-Verzeichnis für Vaults
        self.base_root = vault_root or os.path.join(Config.UPLOAD_FOLDER, 'simulations')
        self._llm_client = None # Lazy loading

    @property
    def llm(self) -> LLMClient:
        # Immer neu erstellen damit aktuelle Config (Modell, Key) genutzt wird
        return LLMClient()

    def _get_vault_path(self, graph_id: str) -> str:
        """Pfad zum spezifischen Vault für diesen Graphen/Simulation"""
        # Wenn graph_id eine simulation_id enthält (z.B. sim_...)
        return os.path.join(self.base_root, graph_id, 'vault')

    def initialize(self, simulation_id: str, graph_name: str = "MiroFish Graph") -> str:
        """Initialisiert die Ordnerstruktur für den Obsidian Vault"""
        vault_path = self._get_vault_path(simulation_id)
        subdirs = ['agents', 'events', 'orgs', 'relations', 'episodes', 'index']
        
        for sd in subdirs:
            os.makedirs(os.path.join(vault_path, sd), exist_ok=True)
            
        # Index-Datei erstellen
        index_file = os.path.join(vault_path, 'index', 'graph_info.json')
        if not os.path.exists(index_file):
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "id": simulation_id,
                    "name": graph_name,
                    "created_at": datetime.now().isoformat(),
                    "provider": "obsidian"
                }, f, indent=2)
                
        return simulation_id # Bei Obsidian ist graph_id = simulation_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """Speichert die Ontologie im Vault"""
        vault_path = self._get_vault_path(graph_id)
        with open(os.path.join(vault_path, 'index', 'ontology.json'), 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)

    def add_text(self, graph_id: str, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Extrahiert Entitäten und Beziehungen aus Text via LLM und speichert sie als Markdown
        (Simuliert Zep's automatische Extraktion)
        """
        vault_path = self._get_vault_path(graph_id)
        
        # 1. Episode speichern
        episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        ep_file = os.path.join(vault_path, 'episodes', f"{episode_id}.md")
        with open(ep_file, 'w', encoding='utf-8') as f:
            f.write(f"---\nid: {episode_id}\ndate: {datetime.now().isoformat()}\n---\n\n{text}")
            
        # Ontologie laden
        ontology = {}
        ontology_file = os.path.join(vault_path, 'index', 'ontology.json')
        if os.path.exists(ontology_file):
            with open(ontology_file, 'r', encoding='utf-8') as f:
                ontology = json.load(f)

        entity_types = [e['name'] for e in ontology.get('entity_types', [])]
        if not entity_types:
            entity_types = ['Person', 'Organization', 'Event', 'Location']

        # LLM-Extraktion
        prompt = f"""Analysiere den folgenden Text und extrahiere alle Entitäten und ihre Beziehungen.

Entitätstypen: {', '.join(entity_types)}

Text:
{text[:3000]}

Antworte NUR mit validem JSON in diesem Format:
{{
  "entities": [
    {{
      "name": "Entitätsname",
      "type": "EinerDerEntitätstypen",
      "summary": "Kurze Beschreibung (1-2 Sätze)",
      "attributes": {{"key": "value"}}
    }}
  ],
  "relations": [
    {{
      "source": "Entitätsname1",
      "target": "Entitätsname2",
      "type": "BEZIEHUNGSTYP",
      "fact": "Beschreibung der Beziehung"
    }}
  ]
}}

Extrahiere 5-15 wichtige Entitäten. Basiere alles auf dem Text."""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192
            )

            logger.debug(f"LLM-Antwort Länge: {len(response)}, Vorschau: {repr(response[:100])}")

            # JSON extrahieren — sowohl roh als auch in Markdown-Codeblock
            cleaned = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*```$', '', cleaned.strip())
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if not json_match:
                logger.warning(f"LLM-Antwort enthält kein JSON: {repr(response[:200])}")
                raise ValueError(f"Kein JSON in LLM-Antwort (Länge {len(response)})")

            extracted = json.loads(json_match.group())

            # Entitäten als Markdown speichern
            entity_name_to_uuid = {}
            for entity in extracted.get('entities', []):
                name = entity.get('name', '').strip()
                if not name:
                    continue

                etype = entity.get('type', 'Person')
                summary = entity.get('summary', '')
                attributes = entity.get('attributes', {})

                safe_name = re.sub(r'[\\/*?:"<>|]', '', name).replace(' ', '_')
                entity_uuid = safe_name
                entity_name_to_uuid[name] = entity_uuid

                folder = 'agents'
                if etype in ('Organization', 'Institution', 'Company', 'MediaOutlet', 'Regulator'):
                    folder = 'orgs'
                elif etype in ('Event', 'Incident', 'Situation'):
                    folder = 'events'

                filepath = os.path.join(vault_path, folder, f"{entity_uuid}.md")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"---\nname: {name}\ntype: {etype}\nuuid: {entity_uuid}\n")
                    f.write(f"episode: {episode_id}\n---\n\n# {name}\n\n**Typ:** {etype}\n\n{summary}\n")
                    if attributes:
                        f.write("\n## Attribute\n")
                        for k, v in attributes.items():
                            f.write(f"- **{k}:** {v}\n")

            # Beziehungen als JSON speichern
            for rel in extracted.get('relations', []):
                src = rel.get('source', '')
                tgt = rel.get('target', '')
                if not src or not tgt:
                    continue

                src_uuid = entity_name_to_uuid.get(src, re.sub(r'[\\/*?:"<>|]', '', src).replace(' ', '_'))
                tgt_uuid = entity_name_to_uuid.get(tgt, re.sub(r'[\\/*?:"<>|]', '', tgt).replace(' ', '_'))
                rel_id = f"rel_{os.urandom(4).hex()}"

                rel_file = os.path.join(vault_path, 'relations', f"{rel_id}.json")
                with open(rel_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "uuid": rel_id,
                        "name": rel.get('type', 'RELATED_TO'),
                        "fact": rel.get('fact', f"{src} -> {tgt}"),
                        "source_node_uuid": src_uuid,
                        "target_node_uuid": tgt_uuid
                    }, f, ensure_ascii=False, indent=2)

            logger.info(f"ObsidianProvider: {len(extracted.get('entities', []))} Entitäten, "
                       f"{len(extracted.get('relations', []))} Beziehungen aus Episode {episode_id} extrahiert")

        except Exception as e:
            logger.warning(f"ObsidianProvider: LLM-Extraktion fehlgeschlagen ({e}), Episode gespeichert")

        return episode_id

    def add_activity(self, graph_id: str, agent_name: str, activity_desc: str):
        """Ergänzt die Aktivität in der Agenten-Datei oder einem globalen Log"""
        vault_path = self._get_vault_path(graph_id)
        
        # Agenten-Dateinamen sanitisieren
        safe_name = re.sub(r'[\\/*?:"<>|]', "", agent_name).replace(" ", "_")
        agent_file = os.path.join(vault_path, 'agents', f"{safe_name}.md")
        
        mode = 'a' if os.path.exists(agent_file) else 'w'
        with open(agent_file, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(f"---\nname: {agent_name}\ntype: agent\n---\n\n# {agent_name}\n\n## Aktivitäten\n")
            f.write(f"- [{datetime.now().strftime('%H:%M:%S')}] {activity_desc}\n")

    def fetch_nodes(self, graph_id: str) -> List[MemoryNode]:
        """Liest alle Markdown-Dateien in agents/, orgs/, events/ und konvertiert sie in Nodes"""
        vault_path = self._get_vault_path(graph_id)
        nodes = []
        
        for folder in ['agents', 'orgs', 'events']:
            folder_path = os.path.join(vault_path, folder)
            if not os.path.exists(folder_path): continue
            
            for file in os.listdir(folder_path):
                if file.endswith('.md'):
                    name = file[:-3].replace("_", " ")
                    nodes.append(MemoryNode(
                        uuid=file[:-3],
                        name=name,
                        labels=[folder.rstrip('s').capitalize()],
                        summary=f"Dokumentierte Entität in {folder}"
                    ))
        return nodes

    def fetch_edges(self, graph_id: str) -> List[MemoryEdge]:
        """Liest Relationen (aktuell rudimentär aus relations/ Ordner)"""
        vault_path = self._get_vault_path(graph_id)
        edges = []
        rel_path = os.path.join(vault_path, 'relations')
        if not os.path.exists(rel_path): return []
        
        for file in os.listdir(rel_path):
            if file.endswith('.json'):
                try:
                    with open(os.path.join(rel_path, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        edges.append(MemoryEdge(**data))
                except Exception:
                    pass
        return edges

    def fetch_node_edges(self, graph_id: str, node_uuid: str) -> List[MemoryEdge]:
        """Filtert alle Kanten für einen bestimmten Knoten"""
        all_edges = self.fetch_edges(graph_id)
        return [e for e in all_edges if e.source_node_uuid == node_uuid or e.target_node_uuid == node_uuid]

    def search(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges") -> MemorySearchResult:
        """Einfache Keyword-Suche in allen Markdown-Dateien"""
        vault_path = self._get_vault_path(graph_id)
        results = []
        query_lower = query.lower()
        
        # Durchsuche Episoden und Agenten
        for root, _, files in os.walk(vault_path):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if query_lower in content.lower():
                            # Einen Ausschnitt als Fakt extrahieren
                            lines = content.split('\n')
                            for line in lines:
                                if query_lower in line.lower() and len(line.strip()) > 10:
                                    results.append(line.strip())
                                    if len(results) >= limit: break
                if len(results) >= limit: break
            if len(results) >= limit: break
            
        return MemorySearchResult(
            query=query,
            facts=results,
            total_count=len(results)
        )

    def get_node_detail(self, graph_id: str, node_uuid: str) -> Optional[MemoryNode]:
        """Liest Details aus der Markdown-Datei"""
        # Suche in allen Ordnern
        vault_path = self._get_vault_path(graph_id)
        for folder in ['agents', 'orgs', 'events']:
            path = os.path.join(vault_path, folder, f"{node_uuid}.md")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return MemoryNode(
                    uuid=node_uuid,
                    name=node_uuid.replace("_", " "),
                    labels=[folder.rstrip('s').capitalize()],
                    summary=content[:500] + "..." if len(content) > 500 else content
                )
        return None

    def delete_graph(self, graph_id: str):
        """Löscht den Vault-Ordner (Vorsicht!)"""
        import shutil
        vault_path = self._get_vault_path(graph_id)
        if os.path.exists(vault_path):
            shutil.rmtree(vault_path)

    def healthcheck(self) -> Dict[str, Any]:
        """Prüft ob das Upload-Verzeichnis beschreibbar ist"""
        try:
            os.makedirs(self.base_root, exist_ok=True)
            return {"status": "ok", "provider": "obsidian", "root": self.base_root}
        except Exception as e:
            return {"status": "error", "provider": "obsidian", "error": str(e)}
