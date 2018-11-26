class Policy_Value:
	def __init__(self, mdp):
		self.mdp = mdp
		"""Initialize values for all states"""
		self.v = [0.0 for i in range(len(mdp.states) + 1)]

		"""Initialize pi: s -> go north"""
		self.pi = dict()
		for state in mdp.states:
			if state in mdp.terminal_states: continue
			self.pi[state] = mdp.actions[0]

	def policy_evaluate(self):
		"""Update value of each state using given policy until converge"""
		
		"""
		for i in range(10000):
			delta = 0.0
			for state in self.mdp.states:
				if state in self.mdp.terminal_states: continue
				action = self.pi[state]
	
				t, s, r = self.mdp.transform(state, action)


				new_v = r + self.mdp.gamma * self.v[s]
				delta += abs(self.v[state] - new_v)
				self.v[state] = new_v

			if delta < 1e-6:
				break
		"""
		delta = 0.0
		for state in self.mdp.states:
			if state in self.mdp.terminal_states: continue
			action = self.pi[state]

			t, s, r = self.mdp.transform(state, action)

			new_v = r + self.mdp.gamma * self.v[s]
			delta += abs(self.v[state] - new_v)
			self.v[state] = new_v

	def policy_improvement(self):
		"""Update policy based on values"""
		for state in self.mdp.states:
			if state in self.mdp.terminal_states: continue
			a1 = self.mdp.actions[0]
			t, s, r = self.mdp.transform(state, a1)
			v1 = r + self.mdp.gamma * self.v[s]

			"""Find a1 that maximizes current values"""
			for action in self.mdp.actions:
				t, s, r = self.mdp.transform(state, action)
				if v1 < r + self.mdp.gamma * self.v[s]:
					a1 = action
					v1 = r + self.mdp.gamma * self.v[s]

			"""Update policy"""
			self.pi[state] = a1


	def value_iteration(self):
		"""Combination of policy evaluation and improvement"""
		for i in range(1000):
			delta = 0.0
			for state in self.mdp.states:
				if state in self.mdp.terminal_states: continue

				a1 = self.pi[state]
				t, s, r = self.mdp.transform(state, a1)
				v1 = r + self.mdp.gamma * self.v[s]

				for action in self.mdp.actions:
					t, s, r = self.mdp.transform(state, action)
					if v1 < r + self.mdp.gamma * self.v[s]:
						a1 = action
						v1 = r + self.mdp.gamma * self.v[s]
					delta += abs(v1 - self.v[s]) 
					self.pi[state] = a1
					self.v[state] = v1
			if delta < 1e-6:
				break









