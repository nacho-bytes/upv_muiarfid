import datetime
import json
import random
import time
import sys
import asyncio

import spade


class PushAgent(spade.agent.Agent):

    async def setup(self):
        self.value = random.randint(1, 1000)

        start_at = datetime.datetime.now() + datetime.timedelta(seconds=5)
        self.add_behaviour(self.PushBehaviour(period=2, start_at=start_at))
        template = spade.template.Template(metadata={"performative": "PUSH"})
        self.add_behaviour(self.RecvBehaviour(), template)

        print("{} ready.".format(self.name))

    def add_value(self, value):
        # seleccion del valor adecuado entre el propio y el nuevo
        self.value = max(self.value, value)

    def add_contacts(self, contact_list):
        self.contacts = [c.jid for c in contact_list if c.jid != self.jid]
        self.length = len(self.contacts)

    # comportamiento encargado de enviar el mensaje push
    class PushBehaviour(spade.behaviour.PeriodicBehaviour):

        async def run(self):
            # el numero de amigos está fijado a 1, se puede modificar
            k=1
            #print("{} period with k={}!".format(self.agent.name, k))
            random_contacts = random.sample(self.agent.contacts, k)
            #print("{} sending to {}".format(self.agent.name, [x.localpart for x in random_contacts]))
            
            # se envia el mensaje con el dato a los k amigos seleccionados
            for jid in random_contacts:
                body = json.dumps({"value": self.agent.value, "timestamp": time.time()})
                msg = spade.message.Message(to=str(jid), body=body, metadata={"performative": "PUSH"})
                await self.send(msg)

    # comportamiento encargado de gestionar la llegada de un mensaje push
    class RecvBehaviour(spade.behaviour.CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=2)
            if msg:
                body = json.loads(msg.body)
                # llamamos al método encargado de decidir si actualiza el dato o no
                self.agent.add_value(body["value"])

                #print("[{}] <{}>".format(self.agent.name, self.agent.value))



async def main(count):
    agents = []
    print("Creating {} agents...".format(count))
    for x in range(1, count + 1):
        print("Creating agent {}...".format(x))
        # nos guardamos la lista de agentes para poder visualizar el estado del proceso gossiping
        # el servidor está fijado a gtirouter.dsic.upv.es, si se tiene un serviodor XMPP en local, se puede sustituir por localhost
        agents.append(PushAgent("push_agent_1626_{}@gtirouter.dsic.upv.es".format(x), "test"))

    # este tiempo trata de esperar que todos los agentes estan registrados, depende de la cantidad de agentes que se lancen
    await asyncio.sleep(3)

    # se le pasa a cada agente la lista de contactos
    for ag in agents:
        ag.add_contacts(agents)
        ag.value = 0

    # se lanzan todos los agentes
    for ag in agents:
        await ag.start()

    # este tiempo trata de esperar que todos los agentes estan ready, depende de la cantidad de agentes que se lancen
    await asyncio.sleep(3)
    
    # este bucle imprime los valores que almacena cada agente y termina cuando todos tienen el mismo valor (consenso)
    while True:
        try:
            await asyncio.sleep(1)
            status = [ag.value for ag in agents]
            print("STATUS: {}".format(status))
            if len(set(status)) <= 1:
                print("Gossip done.")
                break
        except KeyboardInterrupt:
            break

    # se para a todos los agentes
    for ag in agents:
        await ag.stop()
    print("Agents finished")


if __name__ == '__main__':
    count=int(sys.argv[1])
    spade.run(main(count))
