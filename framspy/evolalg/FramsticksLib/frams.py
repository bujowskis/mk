"""Framsticks as a Python module.

Static FramScript objects are available inside the module under their well known names
(frams.Simulator, frams.GenePools, etc.)

These objects and all values passed to and from Framsticks are instances of frams.ExtValue.
Python values are automatically converted to Framstics data types.
Use frams.ExtValue._makeInt()/_makeDouble()/_makeString()/_makeNull() for explicit conversions.
Simple values returned from Framsticks can be converted to their natural Python
counterparts using _value() (or forced to a specific type with  _int()/_double()/_string()).

All non-Framsticks Python attributes start with '_' to avoid conflicts with Framsticks attributes.
Framsticks names that are Python reserved words are prefixed with 'x' (currently just Simulator.ximport).

For sample usage, see frams-test.py and FramsticksLib.py.

If you want to run many independent instances of this class in parallel, use the "multiprocessing" module and then each process
that uses this module will initialize it and get access to a separate instance of the Framsticks library.

For interfaces in other languages (e.g. using the Framsticks library in your C++ code), see ../cpp/frams/frams-objects.h
"""

import ctypes, re, sys, os

c_api = None  # will be initialized in init(). Global because ExtValue uses it.


class ExtValue(object):
	"""All Framsticks objects and values are instances of this class. Read the documentation of the 'frams' module for more information."""

	_reInsideParens = re.compile('\((.*)\)')
	_reservedWords = ['import']  # this list is scanned during every attribute access, only add what is really clashing with Framsticks properties
	_reservedXWords = ['x' + word for word in _reservedWords]
	_encoding = 'utf-8'


	def __init__(self, arg=None, dontinit=False):
		if dontinit:
			return
		if isinstance(arg, int):
			self._initFromInt(arg)
		elif isinstance(arg, str):
			self._initFromString(arg)
		elif isinstance(arg, float):
			self._initFromDouble(arg)
		elif arg == None:
			self._initFromNull()
		else:
			raise ctypes.ArgumentError("Can't make ExtValue from '%s' (%s)" % (str(arg), type(arg)))


	def __del__(self):
		c_api.extFree(self.__ptr)


	def _initFromNull(self):
		self.__ptr = c_api.extFromNull()


	def _initFromInt(self, v):
		self.__ptr = c_api.extFromInt(v)


	def _initFromDouble(self, v):
		self.__ptr = c_api.extFromDouble(v)


	def _initFromString(self, v):
		self.__ptr = c_api.extFromString(ExtValue._cstringFromPython(v))


	@classmethod
	def _makeNull(cls, v):
		e = ExtValue(None, True)
		e._initFromNull()
		return e


	@classmethod
	def _makeInt(cls, v):
		e = ExtValue(None, True)
		e._initFromInt(v)
		return e


	@classmethod
	def _makeDouble(cls, v):
		e = ExtValue(None, True)
		e._initFromDouble(v)
		return e


	@classmethod
	def _makeString(cls, v):
		e = ExtValue(None, True)
		e._initFromString(v)
		return e


	@classmethod
	def _rootObject(cls):
		e = ExtValue(None, True)
		e.__ptr = c_api.rootObject()
		return e


	@classmethod
	def _stringFromC(cls, cptr):
		return cptr.decode(ExtValue._encoding)


	@classmethod
	def _cstringFromPython(cls, s):
		return ctypes.c_char_p(s.encode(ExtValue._encoding))


	def _type(self):
		return c_api.extType(self.__ptr)


	def _class(self):
		cls = c_api.extClass(self.__ptr)
		if cls == None:
			return None
		else:
			return ExtValue._stringFromC(cls)


	def _value(self):
		t = self._type()
		if t == 0:
			return None
		elif t == 1:
			return self._int()
		elif t == 2:
			return self._double()
		elif t == 3:
			return self._string()
		else:
			return self


	def _int(self):
		return c_api.extIntValue(self.__ptr)


	def _double(self):
		return c_api.extDoubleValue(self.__ptr)


	def _string(self):
		return ExtValue._stringFromC(c_api.extStringValue(self.__ptr))


	def _propCount(self):
		return c_api.extPropCount(self.__ptr)


	def _propId(self, i):
		return ExtValue._stringFromC(c_api.extPropId(self.__ptr, i))


	def _propName(self, i):
		return ExtValue._stringFromC(c_api.extPropName(self.__ptr, i))


	def _propType(self, i):
		return ExtValue._stringFromC(c_api.extPropType(self.__ptr, i))


	def _propHelp(self, i):
		h = c_api.extPropHelp(self.__ptr, i)  # unlike other string fields, help is sometimes NULL
		return ExtValue._stringFromC(h) if h != None else '';


	def _propFlags(self, i):
		return c_api.extPropFlags(self.__ptr, i)


	def _propGroup(self, i):
		return c_api.extPropGroup(self.__ptr, i)


	def _groupCount(self):
		return c_api.extGroupCount(self.__ptr)


	def _groupName(self, i):
		return ExtValue._stringFromC(c_api.extGroupName(self.__ptr, i))


	def _groupMember(self, g, i):
		return c_api.extGroupMember(self.__ptr, g, i)


	def _memberCount(self, g):
		return c_api.extMemberCount(self.__ptr, g)


	def __str__(self):
		return self._string()


	def __dir__(self):
		ids = dir(type(self))
		if self._type() == 4:
			for i in range(c_api.extPropCount(self.__ptr)):
				name = ExtValue._stringFromC(c_api.extPropId(self.__ptr, i))
				if name in ExtValue._reservedWords:
					name = 'x' + name
				ids.append(name)
		return ids


	def __getattr__(self, key):
		if key[0] == '_':
			return self.__dict__[key]
		if key in ExtValue._reservedXWords:
			key = key[1:]
		prop_i = c_api.extPropFind(self.__ptr, ExtValue._cstringFromPython(key))
		if prop_i < 0:
			raise AttributeError('no ' + str(key) + ' in ' + str(self))
		t = ExtValue._stringFromC(c_api.extPropType(self.__ptr, prop_i))
		if t[0] == 'p':
			arg_types = ExtValue._reInsideParens.search(t)
			if arg_types:
				arg_types = arg_types.group(1).split(',')  # anyone wants to add argument type validation using param type declarations?


			def fun(*args):
				ext_args = []
				ext_pointers = []
				for a in args:
					if isinstance(a, ExtValue):
						ext = a
					else:
						ext = ExtValue(a)
					ext_args.append(ext)
					ext_pointers.append(ext.__ptr)
				ret = ExtValue(None, True)
				args_array = (ctypes.c_void_p * len(args))(*ext_pointers)
				ret.__ptr = c_api.extPropCall(self.__ptr, prop_i, len(args), args_array)
				return ret


			return fun
		else:
			ret = ExtValue(None, True)
			ret.__ptr = c_api.extPropGet(self.__ptr, prop_i)
			return ret


	def __setattr__(self, key, value):
		if key[0] == '_':
			self.__dict__[key] = value
		else:
			if key in ExtValue._reservedXWords:
				key = key[1:]
			prop_i = c_api.extPropFind(self.__ptr, ExtValue._cstringFromPython(key))
			if prop_i < 0:
				raise AttributeError("No '" + str(key) + "' in '" + str(self) + "'")
			if not isinstance(value, ExtValue):
				value = ExtValue(value)
			c_api.extPropSet(self.__ptr, prop_i, value.__ptr)


	def __getitem__(self, key):
		return self.get(key)


	def __setitem__(self, key, value):
		return self.set(key, value)


	def __len__(self):
		try:
			return self.size._int()
		except:
			return 0


	def __iter__(self):
		class It(object):
			def __init__(self, container, frams_it):
				self.container = container
				self.frams_it = frams_it


			def __iter__(self):
				return self


			def __next__(self):
				if self.frams_it.next._int() != 0:
					return self.frams_it.value
				else:
					raise StopIteration()

		return It(self, self.iterator)


def init(*args):
	"""
	Initializes the connection to Framsticks dll/so/dylib.

	Python programs do not have to know the Framstics path but if they know, just pass the path as the first argument.
	Similarly '-dPATH' and '-DPATH' needed by Framsticks are optional and derived from the first path, unless they are specified as args in init().
	'-LNAME' is the optional library name (full name including the file name extension), default is 'frams-objects.dll/.so' depending on the platform.
	All other arguments are passed to Framsticks and not interpreted by this function.

	"""
	# goals:
	frams_d = None
	frams_D = None
	lib_path = None
	lib_name = ('frams-objects.dylib' if sys.platform == 'darwin' else 'frams-objects.so') if os.name == 'posix' else 'frams-objects.dll'
	initargs = []
	for a in args:
		if a[:2] == '-d':
			frams_d = a
		elif a[:2] == '-D':
			frams_D = a
		elif a[:2] == '-L':
			lib_name = a[2:]
		elif lib_path is None:
			lib_path = a
		else:
			initargs.append(a)
	if lib_path is None:
		# TODO: use environment variable and/or the zip distribution we are in when the path is not specified in arg
		# for now just assume the current dir is Framsticks
		lib_path = '.'

	if os.name == 'nt':
		if sys.version_info < (3, 8):
			original_dir = os.getcwd()
			os.chdir(lib_path)  # because under Windows, frams-objects.dll requires other dll's which reside in the same directory, so we must change current dir for them to be found while loading the main dll.
		else:
			os.add_dll_directory(lib_path)
	abs_data = os.path.join(os.path.abspath(lib_path), "data")  # use absolute path for -d and -D so python is free to cd anywhere without confusing Framsticks

	# for the hypothetical case without lib_path the abs_data must be obtained from somewhere else
	if frams_d is None:
		frams_d = '-d' + abs_data
	if frams_D is None:
		frams_D = '-D' + abs_data
	initargs.insert(0, frams_d)
	initargs.insert(0, frams_D)
	initargs.insert(0, 'dummy.exe')  # as an offset, 0th arg is by convention app name

	global c_api  # access global variable
	if lib_path is not None:  # theoretically, this should only be needed for "and os.name == 'posix'", but in windows python 3.9.5, without using the full lib_name path, there is FileNotFoundError: Could not find module 'frams-objects.dll' (or one of its dependencies). Try using the full path with constructor syntax. Maybe related: https://bugs.python.org/issue42114 and https://stackoverflow.com/questions/59330863/cant-import-dll-module-in-python and https://bugs.python.org/issue39393
		lib_name = os.path.join(lib_path, lib_name)  # lib_path is always set ('.' when not specified). For the (currently unused) case of lib_path==None, the resulting lib_name is a bare filename without any path, which loads the library from a system-configured location.
	try:
		c_api = ctypes.CDLL(lib_name)  # if accessing this module from multiple threads, they will all share a single c_api and access the same copy of the library and its data. If you want separate independent copies, read the comment at the top of this file on using the "multiprocessing" module.
	except OSError as e:
		print("*** Could not find or open '%s' from '%s'.\n*** Did you provide proper arguments and is this file readable?\n" % (lib_name, os.getcwd()))
		raise

	if os.name == 'nt' and sys.version_info < (3, 8):
		os.chdir(original_dir)  # restore current working dir after loading the library so Framsticks sees the expected directory

	c_api.init.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
	c_api.init.restype = None
	c_api.extFree.argtypes = [ctypes.c_void_p]
	c_api.extFree.restype = None
	c_api.extType.argtypes = [ctypes.c_void_p]
	c_api.extType.restype = ctypes.c_int
	c_api.extFromNull.argtypes = []
	c_api.extFromNull.restype = ctypes.c_void_p
	c_api.extFromInt.argtypes = [ctypes.c_int]
	c_api.extFromInt.restype = ctypes.c_void_p
	c_api.extFromDouble.argtypes = [ctypes.c_double]
	c_api.extFromDouble.restype = ctypes.c_void_p
	c_api.extFromString.argtypes = [ctypes.c_char_p]
	c_api.extFromString.restype = ctypes.c_void_p
	c_api.extIntValue.argtypes = [ctypes.c_void_p]
	c_api.extIntValue.restype = ctypes.c_int
	c_api.extDoubleValue.argtypes = [ctypes.c_void_p]
	c_api.extDoubleValue.restype = ctypes.c_double
	c_api.extStringValue.argtypes = [ctypes.c_void_p]
	c_api.extStringValue.restype = ctypes.c_char_p
	c_api.extClass.argtypes = [ctypes.c_void_p]
	c_api.extClass.restype = ctypes.c_char_p
	c_api.extPropCount.argtypes = [ctypes.c_void_p]
	c_api.extPropCount.restype = ctypes.c_int
	c_api.extPropId.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extPropId.restype = ctypes.c_char_p
	c_api.extPropName.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extPropName.restype = ctypes.c_char_p
	c_api.extPropType.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extPropType.restype = ctypes.c_char_p
	c_api.extPropGroup.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extPropGroup.restype = ctypes.c_int
	c_api.extPropFlags.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extPropFlags.restype = ctypes.c_int
	c_api.extPropHelp.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extPropHelp.restype = ctypes.c_char_p
	c_api.extPropFind.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
	c_api.extPropFind.restype = ctypes.c_int
	c_api.extPropGet.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extPropGet.restype = ctypes.c_void_p
	c_api.extPropSet.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
	c_api.extPropSet.restype = ctypes.c_int
	c_api.extPropCall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
	c_api.extPropCall.restype = ctypes.c_void_p
	c_api.extGroupCount.argtypes = [ctypes.c_void_p]
	c_api.extGroupCount.restype = ctypes.c_int
	c_api.extGroupName.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extGroupName.restype = ctypes.c_char_p
	c_api.extGroupMember.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
	c_api.extGroupMember.restype = ctypes.c_int
	c_api.extMemberCount.argtypes = [ctypes.c_void_p, ctypes.c_int]
	c_api.extMemberCount.restype = ctypes.c_int
	c_api.rootObject.argtypes = []
	c_api.rootObject.restype = ctypes.c_void_p

	c_args = (ctypes.c_char_p * len(initargs))(*list(a.encode(ExtValue._encoding) for a in initargs))
	c_api.init(len(initargs), c_args)

	Root = ExtValue._rootObject()
	for n in dir(Root):
		if n[0].isalpha():
			attr = getattr(Root, n)
			if isinstance(attr, ExtValue):
				attr = attr._value()
			setattr(sys.modules[__name__], n, attr)

	print('Using Framsticks version: ' + str(Simulator.version_string))
	print('Home (writable) dir     : ' + home_dir)
	print('Resources dir           : ' + res_dir)
	print()
